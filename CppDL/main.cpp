#include "BaseBlocks/tensor.hpp"

int main(){
    dl::Tensor<1> t1;
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