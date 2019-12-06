% read the image and convert it to doubles
A = double(imread('../sample.jpg'));

for count = 0:100
    tic;
    pure_loops_col_row(A);
    toc
end

for count = 0:100
    tic;
    pure_loops_row_col(A);
    toc
end

matrices_vec(A);
matrices_vec_gpu(A);