clear;clc
close all;

n = 300; % n dimension
r = 5; % r number of column
mu = .8; % mu sparse parameter

epsilon_set = [1e-1;1e-2;1e-3;1e-4;1e-5]; % mu

seed = round(rand() * 10000000);
fprintf('seed:%d\n', seed);
rng(seed);

tab = zeros(length(epsilon_set) * 2, 4);
randnum = 100; 
maxiter = 5000;
outputgap = 10;

for i = 1: length(epsilon_set)
    option.epsilon = epsilon_set(i);
    fail_num_rpnh = 0; success_num_rpnh = 0;
    fail_num_rpncgh = 0; success_num_rpncgh = 0;
    row = (i-1)*2;
    tab(row + 1, 2) = option.epsilon; tab(row + 2, 2) = option.epsilon;
    for j = 1 : randnum
        seed = round(rand() * 10000000);
        fprintf('seed:%d\n', seed);
        rng(seed);
        % generate the random data matrix A
        m = 50;
        A = randn(m,n);
        A = A - repmat(mean(A,1),m,1);
        A = normc(A);
        fprintf('n:%d, r:%d, mu:%f, epsilon:%s, seed:%d\n', n, r, mu, option.epsilon, seed);
        % random initialization
        [phi_init, ~] = svd(randn(n,r),0); 
        x0 = phi_init;

       

        %% Drive_RPN-H
        option.n = n; option.r = r; option.mu = mu;
        option.tol = 1e-8*n*r;
        option.maxiter = maxiter;
        option.x0 = x0;
        option.stop = 1e-10;
        option.outputgap = outputgap;
        [x_rpnH, fs_rpnH, nv_rpnH, iter_rpnH, sparsity_rpnH, time_rpnH, iter_time_rpnH, status] = driver_rpn_h(A, option);
        Fs_rpnH = fs_rpnH(end);
        Ns_rpnH = nv_rpnH(end);
        if status == 1 || iter_rpnH == maxiter
            fail_num_rpnh = fail_num_rpnh + 1;
        end
        if Ns_rpnH <= option.stop
            success_num_rpnh = success_num_rpnh + 1;
        end
      

        %% Drive_RPN-CGH
        [x_rpncgh, fs_rpncgh, nv_rpncgh, iter_rpncgh, sparsity_rpncgh, time_rpncgh, iter_time_rpncgh] = driver_rpn_cgh(A, option);
        Fs_rpncgh = fs_rpncgh(end);
        Ns_rpncgh = nv_rpncgh(end);
        if  iter_rpncgh == maxiter
            fail_num_rpncgh = fail_num_rpncgh + 1;
        end
        if Ns_rpncgh <= option.stop
            success_num_rpncgh = success_num_rpncgh + 1;
        end
  
    end
    tab(row + 1, 3) = fail_num_rpnh;          tab(row + 1, 4) = success_num_rpnh;  
    tab(row + 2, 3) = fail_num_rpncgh;        tab(row + 2, 4) = success_num_rpncgh;  
end

tab(:,1) =[1,2,1,2,1,2,1,2,1,2]; % 1: RPN-H, 2:RPN-CGH
fout = fopen('RPNH_vs_RPNCGH.txt','w');
for i = 1 : size(tab, 1)
    for j = 1 : size(tab, 2)
        if(j == 1)
            fprintf(fout, '%d & ', tab(i, j));
        elseif(j==2)
            fprintf(fout, '$%s$ & ', outputfloat(tab(i, j)));
        else
            fprintf(fout, '%d & ', tab(i, j));
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




