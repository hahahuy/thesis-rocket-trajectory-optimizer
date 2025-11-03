#include <torch/torch.h>
#include <iostream>

int main() {
    torch::Tensor a = torch::ones({2, 3});
    torch::Tensor b = torch::randn({2, 3});
    auto c = a + b;
    std::cout << c << std::endl;
    std::cout << "LibTorch OK" << std::endl;
    return 0;
}

