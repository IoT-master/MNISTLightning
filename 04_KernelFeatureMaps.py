def spatial_size(input_size: int, kernel_size: int, stride: int = 1, padding: int = 0):
    # https://cs231n.github.io/convolutional-networks/
    spatial_size = (input_size - kernel_size + 2 * padding)/stride + 1
    assert spatial_size % 1 == 0
    assert spatial_size > 0
    print(
        f'You will have {spatial_size**2:.0f} feature maps if square dimensions')
    return int(spatial_size)


if __name__ == '__main__':
    print(spatial_size(28, 3))
    # 28 x 28 x 1 => 3 x 3 x 26^2
