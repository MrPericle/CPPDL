# CPPDL - C++ Deep Learning Framework

CPPDL is an open-source framework for deep learning in C++. It aims to provide a comprehensive set of tools and utilities for developing and deploying deep learning models using modern C++ practices. This project is currently a work in progress and welcomes contributions from the community.

## Features

- **Modular Design**: Components are designed to be modular and extendable, allowing for flexibility in model architecture.
- **C++17**: Requires C++17 standard for leveraging modern language features and performance improvements.
- **Tensor Operations**: Efficient tensor operations optimized for performance-critical applications.
- **Work in Progress**: Actively developed with ongoing enhancements and feature additions.

## Getting Started

### Prerequisites

To build and use CPPDL, you need:

- C++17-compatible compiler (e.g., GCC 7.3+, Clang 5.0+, Visual Studio 2019+)



### Example Usage

```
#include <iostream>
#include "cppdl/tensor.h"
#include "cppdl/neural_network.h"

int main(){
    dl::Tensor t1;
    t1.push_back(10);
    t1.push_back(11);
    t1.print();

    dl::Tensor<2> t2D(2);
    t2D.push_back(t1);
    t2D.push_back(t1);

    t2D.print();

    t2D = t2D + t2D;

    t2D.print();

    t2D = t2D * 2; 

    t2D.print();


    return 0;
}
```

## Contributing

We welcome contributions to CPPDL! To contribute:

1. Fork the repository and clone it locally.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push your branch to your fork.
4. Open a pull request against the main repository.

Please ensure that your code adheres to the project's coding style and conventions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions, issues, or suggestions, feel free to [open an issue](https://github.com/mrPericle/CPPDL/issues) or contact the project maintainers directly on periclepergamo99@gmail.com.

## Acknowledgments

- Thanks to all contributors who help make this project better.
- Inspired by the need for a lightweight, efficient deep learning framework in C++.

