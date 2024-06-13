#ifndef CPPDL_ACTIVATIONFUNCTION_LOSSHPP
#define CPPDL_ACTIVATIONFUNCTION_LOSSHPP
#include <vector>
#include <iostream>


class loss_interface{
    public:
        virtual double computeLoss(std::vector<double> vec...) = 0;
};


#endif