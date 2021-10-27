import torch

from src.models import FCNet


def main():
    model = FCNet()
    model.load_state_dict(torch.load('mnist_model_1.pt'))
    model.eval()
    dummy_input = torch.zeros(1, 1, 28, 28)
    torch.onnx.export(model, dummy_input, 'onnx_mnist_1.onnx', verbose=True)

if __name__ == '__main__':
    main()