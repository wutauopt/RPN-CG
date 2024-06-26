clear;clc
close all;

% r = 5; n = 800; mu = 0.8;
% r = 5; n = 1600; mu = 0.8;
% r = 10; n = 800; mu = 0.8;
% r = 5; n = 800; mu = 1;

n_set = [400, 800, 400, 400];
r_set = [8, 8, 12, 8];
mu_set = [0.8, 0.8, 0.8, 1];

% n_set = [200,300,500,1000,1500]; % n dimension
% r = 5; % r rank
% mu = .8; % mu sparse parameter

randnum = 20;
tab = zeros(length(n_set) * 5, 7, randnum); % 5 : number of algorithms
 
maxiter = 5000;
outputgap = 100;

for i = 1:length(n_set)
    n = n_set(i);
    r = r_set(i);
    mu = mu_set(i);
    j = 0; seed = 1; success = 1;
    while(success == 0 || j < 20)
%         seed = round(rand() * 10000000);
        if success == 1
            j = j + 1;
        end
        seed = seed + 1;
        fprintf('seed:%d\n', seed);
        rng(seed);
        % generate the random data matrix A
        m = 50;
        A = randn(m,n);
        A = A - repmat(mean(A,1),m,1);
        A = normc(A);
        fprintf('i:%d, n:%d, r:%d, mu:%f, randnum:%d, seed:%d,\n', i, n, r, mu, j, seed);
        row = (i-1)*5;
        tab(row + 1, 1, j) = n; tab(row + 2, 1, j) = n; 
        tab(row + 3, 1, j) = n; tab(row + 4, 1, j) = n;
        tab(row + 5, 1, j) = n;
        
%         [phi_init, ~] = svd(randn(n,r),0); % random initialization
%         x0 = phi_init;
        
        [U, S, V] = svd(A', 0);
        x0 = U(:, 1 : r);

        %% Drive_ManPG
        option.n = n; option.r = r; option.mu = mu;
        option.tol = 1e-8*n*r;
        option.maxiter = maxiter;
        option.x0 = x0;
        option.stop = 1e-10;
        option.outputgap = outputgap;
        [x_manpg, fs_manpg, nv_manpg, iter_manpg,...
            sparsity_manpg, time_manpg, iter_time_manpg] = driver_ManPG(A, option);
        tab(row + 1, 3, j) = iter_manpg;        tab(row + 1, 4, j) = fs_manpg(end);  
        tab(row + 1, 5, j) = nv_manpg(end);     tab(row + 1, 6, j) = time_manpg;    
        tab(row + 1, 7, j) = sparsity_manpg;

        %% Drive_ManPG_Ada
        [x_manpg_ada, fs_manpg_ada, nv_manpg_ada,...
            iter_manpg_ada, sparsity_manpg_ada, time_manpg_ada,iter_time_manpg_ada] = driver_ManPG_ada(A, option);

        tab(row + 2, 3, j) = iter_manpg_ada;        tab(row + 2, 4, j) = fs_manpg_ada(end);  
        tab(row + 2, 5, j) = nv_manpg_ada(end);     tab(row + 2, 6, j) = time_manpg_ada;    
        tab(row + 2, 7, j) = sparsity_manpg_ada;

        [U, S, V] = svd(x_manpg_ada' * x_manpg);
        if norm(x_manpg_ada - x_manpg * V * U', 'fro') < 1e-2
            sameminimizer = 1;
        else
            success = 0;
            fprintf('ManPG-Ada: converge to different minimizers!\n');
            continue;
        end

        %% ManPQN
        option.type =0;
        option.inner_iter = 200;
        M = 5;
        [X_pqn, F_pqn,F_pqn_list,sp_pqn,t_pqn,...
            maxit_att_pqn,succ_flag_pqn,lins_pqn,in_av_pqn,nv_pqn, iter_time_pqn]= manpqn_orth_sparse(A,option,M,1);

        succ_no_pqn = 1;
        tab(row + 3, 3, j) = sum(maxit_att_pqn)/succ_no_pqn;       tab(row + 3, 4, j) = sum(F_pqn)/succ_no_pqn;  
        tab(row + 3, 5, j) = nv_pqn(end);                          tab(row + 3, 6, j) = sum(t_pqn)/succ_no_pqn;    
        tab(row + 3, 7, j) = sum(sp_pqn)/succ_no_pqn;

        [U, S, V] = svd(X_pqn' * x_manpg);
        if norm(X_pqn - x_manpg * V * U', 'fro') < 1e-2
            sameminimizer = 1;
        else
            success = 0;
            fprintf('ManPQN: converge to different minimizers!\n');
            continue;
        end

        %% Drive_RPN-CG
        [x_rpncg, fs_rpncg, nv_rpncg, iter_rpncg, sparsity_rpncg, time_rpncg, iter_time_rpncg] = driver_rpncg(A, option);
        tab(row + 4, 3, j) = iter_rpncg;        tab(row + 4, 4, j) = fs_rpncg(end);  
        tab(row + 4, 5, j) = nv_rpncg(end);     tab(row + 4, 6, j) = time_rpncg;    
        tab(row + 4, 7, j) = sparsity_rpncg;

        [U, S, V] = svd(x_rpncg' * x_manpg);
        if norm(x_rpncg - x_manpg * V * U', 'fro') < 1e-2
            sameminimizer = 1;
        else
            success = 0;
            fprintf('RPN-CG: converge to different minimizers!\n');
            continue;
        end
        
        %% Drive_RPN-CGH
         option.epsilon = 1e-2;
        [x_rpncgh, fs_rpncgh, nv_rpncgh, iter_rpncgh, sparsity_rpncgh, time_rpncgh, iter_time_rpncgh] = driver_rpn_cgh(A, option);
        tab(row + 5, 3, j) = iter_rpncgh;        tab(row + 5, 4, j) = fs_rpncgh(end); 
        tab(row + 5, 5, j) = nv_rpncgh(end);     tab(row + 5, 6, j) = time_rpncgh;
        tab(row + 5, 7, j) = sparsity_rpncgh;        
         
        [U, S, V] = svd(x_rpncgh' * x_manpg);
        if norm(x_rpncgh - x_manpg * V * U', 'fro') < 1e-2
            sameminimizer = 1;
        else
            success = 0;
            fprintf('RPN-CGH: converge to different minimizers!\n');
            continue;
        end
        success = 1;
    end
end

avetab = mean(tab, 3);
avetab(:,2) =[1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5]; % 1: ManPG, 2: ManPG-Ada,3: ManPQN, 4: RPN-CG, 5:RPN-CGH

fout = fopen('table_rand.txt','w');
for i = 1 : size(avetab, 1)
    for j = 1 : size(avetab, 2)
        if(j == 1 )
           fprintf(fout, '%d & ', round(avetab(i, j))); 
        elseif(j == 2)
            fprintf(fout, '%d & ', avetab(i, j));
        elseif(j == 3)
            fprintf(fout, '%1.2f & ', avetab(i, j));
        elseif(j == size(avetab, 2))
            fprintf(fout, '%1.2f \\\\', avetab(i, j));
        elseif(j == 4 || j == 5)
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



