#ifndef LOSS_H
#define LOSS_H

#include "Tensor.h"

namespace nn {

class Loss {
public:
    virtual ~Loss() = default;
    virtual float compute_loss(const Tensor& predictions, const Tensor& targets) = 0;
    virtual Tensor compute_gradient(const Tensor& predictions, const Tensor& targets) = 0;
};

class MSELoss : public Loss {
public:
    float compute_loss(const Tensor& predictions, const Tensor& targets) override;
    Tensor compute_gradient(const Tensor& predictions, const Tensor& targets) override;
};

class CrossEntropyLoss : public Loss {
public:
    float compute_loss(const Tensor& predictions, const Tensor& targets) override;
    Tensor compute_gradient(const Tensor& predictions, const Tensor& targets) override;
};

} // namespace nn

#endif // LOSS_H