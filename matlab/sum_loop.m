clc;
clear;

total_sum = 0.0;

tic;
for counter = 1:3e9
    total_sum = total_sum + counter;
end
toc
