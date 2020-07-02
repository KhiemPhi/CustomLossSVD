import torch


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
        return input.clamp(min=0)

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

    # Create random Tensors to hold input and outputs.
    x = torch.randn(m, n, device=device, dtype=dtype)

    # Create random Tensors for weights.
    U = torch.randn(m, n, device=device, dtype=dtype, requires_grad=True)
    D = torch.randn(n, device=device, dtype=dtype, requires_grad=True)
    V = torch.randn(n,n, device=device, dtype=dtype, requires_grad=True)

    print(U.shape)
    print(D.shape)
    print(V.shape)

    u,s,v = torch.svd(x)
    print(u.shape)
    print(s.shape)
    print(v.shape)

    print(torch.mm(torch.mm(U, torch.diag(D)), V.t()))

if __name__ == "__main__":
    main()
