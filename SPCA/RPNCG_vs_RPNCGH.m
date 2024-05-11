clear;clc
close all;

n = 1000; % n dimension
r = 10; % r number of column
mu = .8; % mu sparse parameter

seed = round(rand() * 10000000);
fprintf('seed:%d\n', seed);
rng(seed);
% generate the random data matrix A
m = 50;
A = randn(m,n);
A = A - repmat(mean(A,1),m,1);
A = normc(A);

% random initialization
[phi_init, ~] = svd(randn(n,r),0);
x0 = phi_init;

epsilon_set = [1e-1;1e-2;1e-3;1e-4;1e-5]; % mu
randnum = 10; 

tab = zeros(length(epsilon_set) * 2, 7, randnum);

maxiter = 5000;
outputgap = 10;

for i = 1: length(epsilon_set)
    option.epsilon = epsilon_set(i);
    for j = 1 : randnum
        fprintf('n:%d, r:%d, mu:%f, epsilon:%s, seed:%d\n', n, r, mu, option.epsilon, seed);
        row = (i-1)*2;
        tab(row + 1, 2, j) = option.epsilon; tab(row + 2, 2, j) = option.epsilon; 
        
        %% Drive_RPN-CG
        option.n = n; option.r = r; option.mu = mu;
        option.tol = 1e-8*n*r;
        option.maxiter = maxiter;
        option.x0 = x0;
        option.stop = 1e-10;
        option.outputgap = outputgap;
        [x_rpncg, fs_rpncg, nv_rpncg, iter_rpncg, sparsity_rpncg, time_rpncg, iter_time_rpncg] = driver_rpncg(A, option);
        Fs_rpncg = fs_rpncg(end);
        Ns_rpncg = nv_rpncg(end);
        tab(row + 1, 3, j) = iter_rpncg;        tab(row + 1, 4, j) = Fs_rpncg;  
        tab(row + 1, 5, j) = Ns_rpncg;          tab(row + 1, 6, j) = time_rpncg;    
        tab(row + 1, 7, j) = sparsity_rpncg;

        %% Drive_RPN-CGH
        [x_rpncgh, fs_rpncgh, nv_rpncgh, iter_rpncgh, sparsity_rpncgh, time_rpncgh, iter_time_rpncgh] = driver_rpn_cgh(A, option);
        Fs_rpncgh = fs_rpncgh(end);
        Ns_rpncgh = nv_rpncgh(end);
        tab(row + 2, 3, j) = iter_rpncgh;        tab(row + 2, 4, j) = Fs_rpncgh;  
        tab(row + 2, 5, j) = Ns_rpncgh;          tab(row + 2, 6, j) = time_rpncgh;    
        tab(row + 2, 7, j) = sparsity_rpncgh;
    
    end
end

avetab = mean(tab, 3);
avetab(:,1) =[1,2,1,2,1,2,1,2,1,2]; % 1: RPN-CG, 2:RPN-CGH

fout = fopen('RPNCG_vs_RPNCGH.txt','w');
for i = 1 : size(avetab, 1)
    for j = 1 : size(avetab, 2)
        if(j == 1)
            fprintf(fout, '%d & ', avetab(i, j));
        elseif(j == 3)
            fprintf(fout, '%1.2f & ', avetab(i, j));
        elseif(j == size(avetab, 2))
            fprintf(fout, '%1.2f \\\\', avetab(i, j));
        elseif(j==2 ||j == 4 || j == 5)
            fprintf(fout, '$%s$ & ', outputfloat(avetab(i, j)));
        else
            fprintf(fout, '%1.2f & ', avetab(i, j));
        end
    end
    fprintf(fout,'\n');
end
fclose(fout);


function str = outputfloat(x)
    if(x <= 0)
        sn = '-';
        x = abs(x);
    else
        sn = '';
    end
    p = log(x)/log(10);
    p = - ceil(-p);
    x = round(x * 10^(-p) * 100);
    x = x / 100;
    strx = sprintf('%3.2f', x);
    if(p ~= 0)
        str = [sn strx '_{' num2str(p) '}'];
    else
        str = [sn strx];
    end
end



