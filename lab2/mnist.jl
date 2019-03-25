using Printf
using CodecZlib

const train_images = "train-images-idx3-ubyte"
const train_labels = "train-labels-idx1-ubyte"
const test_images = "t10k-images-idx3-ubyte"
const test_labels = "t10k-labels-idx1-ubyte"

const mnist_files = [
    train_images,
    train_labels,
    test_images,
    test_labels,
]

const mnist_urls = [
    "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
    "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
]

mnist_trainset() = (joinpath(@__DIR__, train_images), joinpath(@__DIR__, train_labels))
mnist_testset() = (joinpath(@__DIR__, test_images), joinpath(@__DIR__, test_labels))

function prepare_data()
    global mnist_files, mnist_urls

    for (f, url) ∈ zip(mnist_files, mnist_urls)
        if !isfile(joinpath(@__DIR__, f))
            df = download(url)
            gdf = GzipDecompressorStream(open(df))
            wfd = open(joinpath(@__DIR__, f), "w+")
            write(wfd, gdf)
            close(wfd)
        end
    end

    return @__DIR__
end

process_mnist(filenames::NTuple{2, AbstractString}) = (read_images(filenames[1]), read_labels(filenames[2]))

tolittle(x::Int32) = bswap(x)

function read_images(filename::AbstractString)
    images = open(filename) do f
        magic = tolittle(read(f, Int32))
        @assert magic == 2051

        number = tolittle(read(f, Int32))

        rows = tolittle(read(f, Int32))
        cols = tolittle(read(f, Int32))

        pixels = rows * cols
        images = Vector{Vector{Int}}(undef, number)
        for i = 1:number
            image = Vector{UInt8}(undef, pixels)
            readbytes!(f, image, pixels)
            images[i] = image
        end
        images
    end
end

function read_labels(filename::AbstractString)
    labels = open(filename) do f
        magic = tolittle(read(f, Int32))
        @assert magic == 2049

        number = tolittle(read(f, Int32))

        labels = Vector{Int}(undef, number)
        for i = 1:number
            labels[i] = read(f, UInt8)
        end
        labels
    end
end

function binary_plot(image::AbstractVector{Int}, image_size=(28, 28))
    row, col = image_size
    for i = 1:length(image)
        v = image[i]::Int
        @assert 0 <= v <= 255
        if 0 <= v <= 127
            print("0 ")
        else
            print("1 ")
        end

        if i % col == 0
            println()
        end
    end
    image
end

function number_plot(image::AbstractVector{Int}, image_size=(28, 28))
    row, col = image_size
    for i = 1:length(image)
        v = image[i]::Int
        @assert 0 <= v <= 255

        Printf.@printf "%3d " v

        if i % col == 0
            println()
        end
    end
    image
end
