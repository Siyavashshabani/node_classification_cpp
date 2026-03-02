# Dynamic-Head GCN (LibTorch C++)

## Environment (my setup)
- OS: Ubuntu 22.04
- Compiler: GCC 11.4.0
- LibTorch: 2.11.0
- CUDA (`nvidia-smi`): 12.9
- CUDA Toolkit (nvcc): 12.8

## How to run the code: 

```bash
cmake --build build -j
./build/graph_node_classifaction_cpp
```

## How dynamic heads were registered (naming scheme)
Each head is created as a `torch::nn::Linear(hidden_dim, out_dim)` and registered as a submodule using a stable name:

- Module name: `head_<name>` (e.g., `head_cls`, `head_cls2`)
- Registration call: `register_module("head_" + name, head)`

Heads are also stored for fast lookup in:
`std::unordered_map<std::string, torch::nn::Linear> heads;`

Because each head is registered as a submodule, its parameters appear in `named_parameters()` and are therefore included automatically in optimizers and checkpoint saving/loading.


## What happened when loading a checkpoint into a model with an extra head Workflow?
The checkpoint was saved from a model that contained the trunk (Two GCN layers) and one head (`cls`, out_dim=3).  
When loading into a new model that also has an extra head (`cls2`, out_dim=2), parameters are matched by their registered module names (e.g., `W0.*`, `W1.*`, `head_cls.*`).

As a result:
- The trunk weights(Two GCN layers) were loaded successfully.
- The `cls` head weights were loaded successfully because the module name and tensor shapes match.
- The extra head `cls2` was not present in the checkpoint, so its weights were not loaded and remained at their original random initialization. This was verified by comparing the `cls2` weight norm before and after loading:

  - `cls2` weight norm (before load): 0.899028  
  - `cls2` weight norm (after load):  0.899028

Although the checkpoints are loaded without any error, this is not safe for deployment/production, since cls2 is not trained. As a warning we are printing the
Missing keys(cls2):
- `head_cls2.weight`
- `head_cls2.bias`

## One improvement for production safety
As a way to improve the reliability of the project, we can add a schema_version and model metadata to each checkpoint; the loader for the new model validates compatibility and refuses to load when the checkpoint structure doesn’t match, preventing silent partial weight loading.

## Optional stretch (bonus points)
To address this section, synthetic data is generated where each node has three different input groups: `inputs["feat6"]`, `inputs["feat12"]`, and `inputs["feat8"]`. Other parameters, such as the number of nodes, are the same as in the previous synthetic dataset. The `add_input_group` function acts as an adapter that maps each input group to a target-dimensional representation (`Linear(in_dim, in_F)`). These adapted inputs are then combined using the `build_X_from_inputs` function with either `"mean"` or `"sum"`.

All other parts of the code are the same as in the previous file. To keep the main implementation clean, the optional section is implemented in `main_opt.cpp`.

### How to run (optional)
1. In `CMakeLists.txt`, uncomment:
   - `add_executable(graph_node_classifaction_cpp src/main_opt.cpp)`
2. Build and run:
```bash
cmake --build build -j
./build/graph_node_classifaction_cpp
```

## References: 
1- `https://docs.alcf.anl.gov/aurora/data-science/inference/libtorch/`

2- `https://docs.pytorch.org/tutorials/advanced/cpp_frontend.html`

3- `https://arxiv.org/pdf/1609.02907`
