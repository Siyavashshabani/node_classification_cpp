#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <cmath>

//------------------------------------------------------  GCN model 
struct GCNImpl : torch::nn::Module {
    torch::nn::Linear W0{nullptr};
    torch::nn::Linear W1{nullptr};
    int64_t in_F;
    int64_t hidden_dim;

    std::unordered_map<std::string, torch::nn::Linear> heads;
    std::unordered_map<std::string, torch::nn::Linear> adapters;

    GCNImpl(int64_t F, int64_t hidden) : in_F(F), hidden_dim(hidden) {
        W0 = register_module("W0", torch::nn::Linear(in_F, hidden_dim));
        W1 = register_module("W1", torch::nn::Linear(hidden_dim, hidden_dim));
    }

    torch::Tensor trunk_forward(const torch::Tensor& A_hat, const torch::Tensor& X) {
        auto H1 = torch::matmul(A_hat,W0->forward(X));
        H1 = torch::relu(H1);
        auto H2 = torch::matmul(A_hat, W1->forward(H1));
        H2 = torch::relu(H2);
        return H2;
    }

    void add_head(const std::string& name, int64_t out_dim) {
        if (heads.count(name)) {
            std::cout << "[add_head] already exists: " << name << "\n";
            return;
        }
        auto head = torch::nn::Linear(hidden_dim, out_dim);
        head->to(W0->weight.device());
        register_module("head_" +name, head);
        heads.emplace(name, head);
        std::cout << "[add_head] added head_"<< name << ":"<< hidden_dim <<"->"<<out_dim<<"\n";
    }

    void add_input_group(const std::string& name, int64_t in_dim) {
        auto adapter = torch::nn::Linear(in_dim, in_F);
        adapter->to(W0->weight.device());

        register_module("adapter_" + name, adapter);
        adapters.emplace(name, adapter);

        std::cout << "[add_input_group] added adapter_" << name << " : " << in_dim << " -> " << in_F << "\n";
    }

    void remove_input_group(const std::string& name) {
        auto it = adapters.find(name);
        if (it == adapters.end()) {
            std::cout << "[remove_input_group] not found: " << name << "\n";
            return;
        }
        adapters.erase(it);
        std::cout << "[remove_input_group] removed (inactive) adapter: " << name << "\n";
    }

    torch::Tensor build_X_from_inputs(
        const std::unordered_map<std::string, torch::Tensor>& inputs,
        const std::vector<std::string>& active_inputs,
        const std::string& combine = "mean"
    ) {
        if (active_inputs.empty()) {
            throw std::runtime_error("active_inputs is empty");
        }

        torch::Tensor X_acc;
        int64_t N_ref = -1;

        for (size_t i = 0; i < active_inputs.size(); ++i) {
            const auto& name = active_inputs[i];

            auto it_in = inputs.find(name);
            if (it_in == inputs.end()) {
                throw std::runtime_error("Input group not found in inputs: " + name);
            }

            auto it_ad = adapters.find(name);
            if (it_ad == adapters.end()) {
                throw std::runtime_error("Adapter not found for input group: " + name);
            }

            const auto& Xin = it_in->second;
            if (Xin.dim() != 2) {
                throw std::runtime_error("Input must be rank-2 [N, in_dim] for group: " + name);
            }

            int64_t N = Xin.size(0);
            if (N_ref < 0) N_ref = N;
            if (N != N_ref) {
                throw std::runtime_error("All input groups must have same N. Mismatch at: " + name);
            }

            auto Xproj = it_ad->second->forward(Xin);
            if (i == 0) X_acc = Xproj;
            else X_acc = X_acc + Xproj;
        }

        if (combine == "mean") {
            X_acc = X_acc / static_cast<double>(active_inputs.size());
        } else if (combine != "sum") {
            throw std::runtime_error("Unknown combine mode: " + combine);
        }

        return X_acc;
    }

    std::unordered_map<std::string, torch::Tensor>
    forward(const torch::Tensor& A_hat,
            const torch::Tensor& X,
            const std::vector<std::string>& active_heads) {

        auto H2 = trunk_forward(A_hat, X);

        std::unordered_map<std::string, torch::Tensor> outputs;
        outputs.reserve(active_heads.size());

        for (const auto& name : active_heads) {
            auto it = heads.find(name);
            if (it == heads.end()) {
                throw std::runtime_error("Active head not found: " + name);
            }
            outputs.emplace(name, it->second->forward(H2));
        }
        return outputs;
    }

    std::unordered_map<std::string, torch::Tensor>
    forward(const torch::Tensor& A_hat,
            const std::unordered_map<std::string, torch::Tensor>& inputs,
            const std::vector<std::string>& active_inputs,
            const std::vector<std::string>& active_heads,
            const std::string& combine = "mean") {

        auto X = build_X_from_inputs(inputs, active_inputs, combine);
        return forward(A_hat, X, active_heads);
    }
};
TORCH_MODULE(GCN);


//------------------------------------------------------  Adjacency Matrix 

static torch::Tensor normalize_adj(const torch::Tensor& A) {
    const auto N = A.size(0);
    auto I = torch::eye(N, A.options());
    auto A_tilde = A + I;
    auto deg = A_tilde.sum(1);                
    auto deg_inv_sqrt = torch::pow(deg, -0.5);

    deg_inv_sqrt = torch::where(torch::isfinite(deg_inv_sqrt),
                                deg_inv_sqrt,
                                torch::zeros_like(deg_inv_sqrt));
    return deg_inv_sqrt.unsqueeze(1) * A_tilde * deg_inv_sqrt.unsqueeze(0);
}

int main() {
    torch::manual_seed(0);

    // ----- Config
    const int64_t N = 64;
    const int64_t F = 8;
    const int64_t hidden = 16;
    const double p = 0.5;
    const double noise_std = 0.01;

    // ----- define the device
    torch::Device device(torch::kCUDA); 
    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(device);

    //------------------------------------------------------ dataset 
    // -----  A 
    auto R = torch::rand({N, N}, opts_f);
    auto upper = torch::triu((R < p).to(torch::kFloat32), 1);
    auto A = upper + upper.transpose(0, 1);
    auto A_hat = normalize_adj(A);

    // ----- Node features
    std::unordered_map<std::string, torch::Tensor> inputs;
    inputs["feat6"]= torch::randn({N,6}, opts_f);
    inputs["feat12"] = torch::randn({N,12},opts_f);
    inputs["feat8"]= torch::randn({N,8},opts_f);
    auto X = inputs.at("feat8");

    // ----- Labels y (3 classes) Example: [0.2, -1.1, 0.5] max will be: 0.5 then the label is 2  
    auto tmp_logits = X.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)})
                    + noise_std * torch::randn({N, 3}, opts_f);
    auto y = std::get<1>(tmp_logits.max(1)).to(torch::kLong).to(device); // [N]


    std::cout << "dataset checking-------------------------------------------------------"<< "\n";
    std::cout << "A_hat shape----:"<< A_hat.sizes() << "\n";
    std::cout << "y shape--------:"<< y.sizes() <<"\n";
    std::cout << "feat6  shape--------: " << inputs.at("feat6").sizes()<< "\n";
    std::cout << "feat12 shape------: " << inputs.at("feat12").sizes()<< "\n";
    std::cout << "feat8  shape-------: " << inputs.at("feat8").sizes() << "\n";

    //------------------------------------------------------ model  
    std::cout << "Defining the GCN model model--------------------------------------------"<< "\n";
    auto model = GCN(F, hidden);
    model->to(device);
    model->train();
    model->add_head("cls", 3);
    model->add_input_group("feat6", 6);
    model->add_input_group("feat12", 12);
    model->add_input_group("feat8", 8);

    torch::optim::Adam opt(model->parameters(), torch::optim::AdamOptions(5e-1));

    const std::vector<std::string> active_heads = {"cls"};
    const std::vector<std::string> active_inputs = {"feat6", "feat12", "feat8"};

    std::cout << "Start training loop----------------------------------------------------" << "\n";
    for (int step = 1; step <= 100; ++step) {
        opt.zero_grad();

        auto outputs = model->forward(A_hat, inputs, active_inputs, active_heads, "mean");
        auto logits = outputs.at("cls"); 
        auto loss = torch::nn::functional::cross_entropy(logits, y);

        loss.backward();
        opt.step();

        if (step % 5 == 0) {
            float loss_val = loss.item<float>();
            std::cout << "step " << step << " | loss = " << loss_val << "\n";
        }
    }

    return 0;
}