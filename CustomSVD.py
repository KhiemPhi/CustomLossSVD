import torch
import torch.nn as nn


class MySVD(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        """
                In the forward pass we receive a Tensor containing the input and return
                a Tensor containing the output. ctx is a context object that can be used
                to stash information for backward computation. You can cache arbitrary
                objects for use in the backward pass using the ctx.save_for_backward method.
        """

        ctx.save_for_backward(input)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        """
                In the backward pass we receive a Tensor containing the gradient of the loss
                with respect to the output, and we need to compute the gradient of the loss
                with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()

        return grad_input


def main():
    dtype = torch.float
    device = torch.device("cpu")

    m = 125
    n = 100

    max = 0.25
    min = 0.0001

    # Create random Tensors to hold input and outputs.
    x = torch.randn(m, n, device=device, dtype=dtype)

    # Create random Tensors for weights.
    #U, D, V = torch.svd(x)

    U = (max-min)*torch.randn(m, n, device=device, dtype=dtype) - min
    D = (max-min)*torch.randn(n, device=device, dtype=dtype) - min
    V = (max-min)*torch.randn(n, n, device=device, dtype=dtype) - min

    U.requires_grad = True
    D.requires_grad = True
    V.requires_grad = True

    learning_rate = 1e-25

    for t in range(100):  # training for 10 iterations
        # To apply our function, we use Function.apply method
        svd = MySVD.apply

        x_pred = svd(torch.mm(torch.mm(U, torch.diag(D)), V.t()))

        # SSE
        loss = torch.dist(x, x_pred)

        # MAE
        #loss = torch.abs(x_pred -x).mean()

        if t % 1 == 0:
            print(t, loss.item())

        loss.backward()

        # Update weights using gradient descent
        with torch.no_grad():
            U -= learning_rate * U.grad
            D -= learning_rate * D.grad
            V -= learning_rate * V.grad

            # Manually zero the gradients after updating weights
            U.grad.zero_()
            D.grad.zero_()
            V.grad.zero_()

    U.requires_grad = False
    D.requires_grad = False
    V.requires_grad = False

    u, s, v = torch.svd(x)
    torch_svd_result = torch.mm(torch.mm(u, torch.diag(s)), v.t())
    custom_svd_result = torch.mm(torch.mm(U, torch.diag(D)), V.t())

    loss_torch = torch.abs(torch_svd_result -x).mean()
    loss_custom = torch.abs(custom_svd_result -x).mean()

    print(loss_torch)
    print(loss_custom)


if __name__ == "__main__":
    main()
