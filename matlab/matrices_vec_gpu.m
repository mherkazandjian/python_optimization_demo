function A = matrices_vec_gpu(A)

    A11 = gpuArray(circshift(A,-1,1));
    A12 = gpuArray(circshift(A,1,1));
    A21 = gpuArray(circshift(A,-1,2));
    A22 = gpuArray(circshift(A,1,2));
    
    for count = 0:1000
        tic;
        A = -4.0*A + A11 + A12 + A21 + A22;
        toc
    end
end