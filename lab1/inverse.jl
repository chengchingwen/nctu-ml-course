Lij(LU::AbstractMatrix{T}, i::Int, j::Int) where T = i < j ? zero(T) : @inbounds LU[i, j]::T
function Lij(LU::M, i::Int) where M
    n = size(LU, 1)
    Lij(LU, rem(i-1, n) + 1, div(i-1, n) + 1)
end

Uij(LU::AbstractMatrix{T}, i::Int, j::Int) where T = i > j ? zero(T) : i == j ? one(T) : @inbounds LU[i, j]::T
function Uij(LU::M, i::Int) where M
    n = size(LU, 1)
    Uij(LU, rem(i-1, n) + 1, div(i-1, n) + 1)
end

function LUDecomp(A::AbstractMatrix{T}) where T
    LU = LUDecomp_core(A)
    getL(LU), getU(LU)
end

function LUDecomp_core(A::AbstractMatrix{T}) where T
    LU = copy(A)
    n = size(A, 1)
    @inbounds for i = 2:n
        LU[1, i] = A[1, i] / A[1, 1]
    end

    @inbounds for j = 2:n
        for i = j:n
            for k = 1:j-1
                LU[i, j] -= Lij(LU, i, k) * Uij(LU, k, j)
            end
        end

        for i = j+1:n
            for k = 1:j-1
                LU[j, i] -= Lij(LU, j, k) * Uij(LU, k, i)
            end
            LU[j, i] /= LU[j, j]
        end
    end
    LU
end

function getL(LU::AbstractMatrix{T}) where T
    L = zeros(size(LU))
    n = size(LU, 1)
    for ind = 1:length(L)
        if rem(ind-1, n) >= div(ind-1, n)
            @inbounds L[ind] = LU[ind]
        end
    end
    L
end

function getU(LU::AbstractMatrix{T}) where T
    U = one(LU)
    n = size(LU, 1)
    for ind = 1:length(U)
        if rem(ind-1, n) < div(ind-1, n)
            @inbounds U[ind] = LU[ind]
        end
    end
    U
end

function inverse_L(LU::AbstractMatrix{T}) where T
    Linv = one(LU)
    n = size(LU, 1)
    @inbounds for j = 1:n
        Linv[j, j] /= LU[j, j]
        for i = j+1:n
            for k = j:i-1
                Linv[i, j] -= LU[i, k] * Linv[k, j]
            end
            Linv[i, j] /= LU[i, i]
        end
    end
    Linv
end

function inverse_U(LU::AbstractMatrix{T}) where T
    Uinv = one(LU)
    n = size(LU, 1)
    @inbounds for j = 1:n
        for i = j+1:n
            for k = j:i-1
                Uinv[j, i] -= Uinv[j, k] * LU[k, i]
            end
        end
    end
    Uinv
end

function inverse_LU(LU::AbstractMatrix{T}) where T
    n = size(LU, 1)
    @inbounds for j = 1:n
        LU[j, j] = one(T) / LU[j, j]
        for i = j+1:n
            ltmp = utmp = zero(T)
            for k = j:i-1
                ltmp -= LU[i, k] * LU[k, j]
                if j == k
                    utmp -= LU[k, i]
                elseif i == k
                    utmp -= LU[j, k]
                else
                    utmp -= LU[j, k] * LU[k, i]
                end
            end
            LU[i, j] = ltmp / LU[i, i]
            LU[j, i] = utmp
        end
    end
    LU
end

function multiply_UL(LU::AbstractMatrix{T}) where T
    invA = zeros(T, size(LU))
    n = size(LU, 1)
    @inbounds for i = 1:n
        for j = 1:n
            for k = j:n
                invA[i, j] += Uij(LU, i, k) * Lij(LU, k, j)
            end
        end
    end
    invA
end

function inverse(A::AbstractMatrix{T}) where T
    LU = LUDecomp_core(A)
    inverse_LU(LU)
    multiply_UL(LU)
end
