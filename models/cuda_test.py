import torch


def main():
    print(f'device name [0]:', torch.cuda.get_device_name(0))
    print(torch.cuda.is_available())

if __name__ == '__main__':
    main()
