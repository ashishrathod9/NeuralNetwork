#ifndef LOSS_H
#define LOSS_H

#include "Tensor.h"

namespace nn {

class Loss {
public:
    virtual ~Loss() = default;
    virtual float compute(const Tensor& predicted, const Tensor& actual) = 0;
    virtual Tensor compute_gradient(const Tensor& predicted, const Tensor& actual) = 0;
};

class MSELoss : public Loss {
public:
    float compute(const Tensor& predicted, const Tensor& actual) override;
    Tensor compute_gradient(const Tensor& predicted, const Tensor& actual) override;
};

} // namespace nn

#endif // LOSS_H