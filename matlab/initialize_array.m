clc;
clear;

x1 = magic(10000);
x2 = magic(10000);

for c = 0:10
    tic;
    x3 = x1 .* x2;
    sum(sum(x3));
    toc
end

g1 = gpuArray(x1);
g2 = gpuArray(x2);

for c = 0:10000
    tic;
    g3 = g1 .* g2;
    sum(sum(g3));
    toc
end

