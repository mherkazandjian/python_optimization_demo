function A = matrices_vec(A)

    A11 = circshift(A,-1,1);
    A12 = circshift(A,1,1);
    A21 = circshift(A,-1,2);
    A22 = circshift(A,1,2);
      
    for count = 0:100
        tic;
        A = -4.0*A + A11 + A12 + A21 + A22;
        toc
    end
end