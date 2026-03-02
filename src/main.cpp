#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <cmath>


static void save_key_manifest(const torch::nn::Module& model, const std::string& path_txt) {
    std::ofstream f(path_txt);
    if (!f) throw std::runtime_error("Failed to open manifest file: " + path_txt);
    for (const auto& p : model.named_parameters(/*recurse=*/true)) {
        f << p.key() << "\n";
    }
}

static std::unordered_set<std::string> read_key_manifest(const std::string& path_txt) {
    std::ifstream f(path_txt);
    if (!f) throw std::runtime_error("Failed to open manifest file: " + path_txt);
    std::unordered_set<std::string> keys;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty()) keys.insert(line);
    }
    return keys;
}

static void save_state_dict(const torch::nn::Module& model, const std::string& path) {
    torch::serialize::OutputArchive archive;
    auto params = model.named_parameters(/*recurse=*/true);
    for (const auto& param : params) {
        archive.write(param.key(), param.value());
    }
    archive.save_to(path);
}

static void load_state_dict(
    torch::nn::Module& model,
    const std::string& ckpt_pt,
    const std::string& manifest_txt,
    std::vector<std::string>& missing_keys,
    std::vector<std::string>& unexpected_keys
) {
    // load archive
    torch::serialize::InputArchive archive;
    archive.load_from(ckpt_pt);

    // checkpoint keys from manifest
    auto ckpt_keys = read_key_manifest(manifest_txt);

    // model keys
    std::unordered_set<std::string> model_keys;
    for (const auto& p : model.named_parameters(/*recurse=*/true)) {
        model_keys.insert(p.key());
    }

    // missing = in model but not in ckpt
    for (const auto& k : model_keys) {
        if (!ckpt_keys.count(k)) missing_keys.push_back(k);
    }

    // unexpected = in ckpt but not in model
    for (const auto& k : ckpt_keys) {
        if (!model_keys.count(k)) unexpected_keys.push_back(k);
    }

    // load only matching keys
    for (auto& p : model.named_parameters(/*recurse=*/true)) {
        const std::string& k = p.key();
        if (!ckpt_keys.count(k)) continue;

        torch::Tensor t;
        try {
            archive.read(k, t);
        } catch (...) {
            continue;
        }

        if (!t.defined() || t.sizes() != p.value().sizes()) {
            missing_keys.push_back(k + " (shape mismatch)");
            continue;
        }

        {
            torch::NoGradGuard ng;
            p.value().copy_(t);
        }
    }
}

static void print_parameter_names(torch::nn::Module& m, const std::string& title) {
    std::cout << "\n==== " << title << " ====\n";
    for (const auto& p : m.named_parameters(/*recurse=*/true)) {
        std::cout << p.key() << "  shape=" << p.value().sizes() << "\n";
    }
}


//------------------------------------------------------  GCN model 
struct GCNImpl : torch::nn::Module {
    torch::nn::Linear W0{nullptr};
    torch::nn::Linear W1{nullptr};
    int64_t hidden_dim;
    std::unordered_map<std::string, torch::nn::Linear> heads;
    GCNImpl(int64_t F, int64_t hidden) : hidden_dim(hidden) {
        W0 = register_module("W0", torch::nn::Linear(F, hidden_dim));
        W1 = register_module("W1", torch::nn::Linear(hidden_dim, hidden_dim));
    }

    torch::Tensor trunk_forward(const torch::Tensor& A_hat, const torch::Tensor& X) {
        auto H1 = torch::matmul(A_hat, W0->forward(X));
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

        // Move head to same device as trunk parameters
        head->to(W0->weight.device());

        register_module("head_" + name, head);
        heads.emplace(name, head);

        std::cout << "[add_head] added head_" << name << " : " << hidden_dim << " -> " << out_dim << "\n";
    }

    void remove_head(const std::string& name) {
        auto it = heads.find(name);
        if (it == heads.end()) {
            std::cout << "[remove_head] not found: " << name << "\n";
            return;
        }
        heads.erase(it);
        std::cout << "[remove_head] removed (inactive) head: " << name << "\n";
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
            outputs.emplace(name, it->second->forward(H2)); // [N, out_dim]
        }
        return outputs;
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


//------------------------------------------------------  Main code
int main() {
    torch::manual_seed(0);

    // ----- Config
    const int64_t N = 64;
    const int64_t F = 8;
    const int64_t hidden = 16;
    const double p = 0.5;
    const double noise_std = 0.01;

    // ----- define the device 
    torch::Device device(torch::kCUDA);  // or kCPU
    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(device);

    //------------------------------------------------------ dataset 
    // ----- Build A 
    auto R = torch::rand({N, N}, opts_f);
    auto upper = torch::triu((R < p).to(torch::kFloat32), 1);
    auto A = upper + upper.transpose(0, 1);
    auto A_hat = normalize_adj(A);

    // ----- Node features
    auto X = torch::randn({N, F}, opts_f);

    // ----- Labels y (3 classes) Example: [0.2, -1.1, 0.5] max will be: 0.5 then the label is 2  
    auto tmp_logits = X.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)})
                    + noise_std * torch::randn({N, 3}, opts_f);
    auto y = std::get<1>(tmp_logits.max(1)).to(torch::kLong).to(device); // [N]


    std::cout << "dataset checking-------------------------------------------------------"<< "\n";
    std::cout << "A_hat shape----:" << A_hat.sizes() << "\n";
    std::cout << "X shape--------:" << X.sizes() << "\n";
    std::cout << "y shape--------:" << y.sizes() << "\n";

    //------------------------------------------------------ model  
    std::cout << "Defining the GCN model model--------------------------------------------"<< "\n";
    auto model = GCN(F, hidden);
    model->to(device);
    model->train();
    print_parameter_names(*model, "PARAMETERS BEFORE adding cls");
    std::cout << "\n";

    model->add_head("cls", 3);

    print_parameter_names(*model, "PARAMETERS AFTER adding cls");
    std::cout << "\n";

    torch::optim::Adam opt(model->parameters(), torch::optim::AdamOptions(1e-1));

    const std::vector<std::string> active_heads = {"cls"};

    std::cout << "Start training loop----------------------------------------------------" << "\n";
    for (int step = 1; step <= 30; ++step) {
        opt.zero_grad();

        auto outputs = model->forward(A_hat, X, active_heads);
        auto logits = outputs.at("cls"); 

        auto loss = torch::nn::functional::cross_entropy(logits, y);

        loss.backward();
        opt.step();

        if (step % 5 == 0) {
            std::cout << "step " << step << " | loss = " << loss.item<float>() << "\n";
        }
    }

    //------------------------------------------------------ saving the weights
    std::cout << "Save the checkpoints--------------------------------------------------" << "\n";
    const std::string ckpt_dir = "./saved_weights";
    const std::string ckpt_path = ckpt_dir + "/gcn_ckpt.pt";
    const std::string manifest_path = ckpt_dir + "/gcn_ckpt_keys.txt";

    std::string cmd = "mkdir -p " + ckpt_dir;
    int rc = std::system(cmd.c_str());
    if (rc != 0) {
        std::cerr << "mkdir failed (rc=" << rc << "): " << cmd << "\n";
    }

    save_state_dict(*model, ckpt_path);
    save_key_manifest(*model, manifest_path);
    std::cout << "Saved checkpoint to: " << ckpt_path << "\n";
    std::cout << "Saved key manifest to: " << manifest_path << "\n";

    //------------------------------------------------------ laoding the saved weights
    std::cout << "Creating new head(cls2)--------------------------------------------" << "\n";
    auto model2 = GCN(F, hidden);
    model2->to(device);
    model2->add_head("cls", 3);
    model2->add_head("cls2", 2);

    auto cls2_w_before = model2->heads.at("cls2")->weight.norm().item<float>();
    std::cout << "cls2 weight norm (before load): " << cls2_w_before << "\n";

    std::cout << "Loading saved weights------------------------------------------------" << "\n";
    std::vector<std::string> missing, unexpected;
    load_state_dict(*model2, ckpt_path, manifest_path, missing, unexpected);

    auto cls2_w_after = model2->heads.at("cls2")->weight.norm().item<float>();
    std::cout << "cls2 weight norm (after load):  " << cls2_w_after << "\n";


    std::cout << "\n=== Warning for loaded checkpoints ===\n";
    std::cout << "Missing keys (" << missing.size() << "):\n";
    for (const auto& k : missing) std::cout << "  - " << k << "\n";
    std::cout << "Unexpected keys (" << unexpected.size() << "):\n";
    for (const auto& k : unexpected) std::cout << "  - " << k << "\n";


    model2->eval();
    {
        torch::NoGradGuard ng;
        const std::vector<std::string> heads_run = {"cls", "cls2"};
        auto out = model2->forward(A_hat, X, heads_run);
        std::cout << "\nForward OK.\n";
        std::cout << "cls  logits shape:  " << out.at("cls").sizes()  << "\n"; 
        std::cout << "cls2 logits shape: " << out.at("cls2").sizes() << "\n";  
    }

    // sanity: cls2 should not change by loading (norms equal)
    if (std::abs(cls2_w_before - cls2_w_after) < 1e-6) {
        std::cout << "Confirmed: cls2 was NOT loaded (weights unchanged).\n";
    } else {
        std::cout << "Warning: cls2 weights changed (unexpected).\n";
    }

    return 0;
}