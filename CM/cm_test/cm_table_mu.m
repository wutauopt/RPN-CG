clear;clc
close all;

mu_set = [0.15;0.25;0.35;0.45]; 

n = 512; % n dimension
r = 5;
randnum = 2;
tab = zeros(length(mu_set) * 5, 7, randnum); % 4 : number of algorithms
 
maxiteration = 3000;
outputgap = 10;

for i = 1:length(mu_set)
    mu = mu_set(i);
    for j = 1 : randnum
        seed = round(rand() * 10000000);
        fprintf('seed:%d\n', seed);
        rng(seed);
        % generate the random data matrix A
        L = 50; dx = L/n;  V = 0;
        H = -Sch_matrix(0,L,n);
        Hmaxsigma = svds(H, 1);
        fprintf('n:%d, r:%d, mu:%f, randnum:%d, seed:%d\n', n, r, mu, j, seed);
        row = (i-1)*5;
        tab(row + 1, 1, j) = mu; tab(row + 2, 1, j) = mu; 
        tab(row + 3, 1, j) = mu; tab(row + 4, 1, j) = mu;
        tab(row + 5, 1, j) = mu;
        
        [phi_init, ~] = svd(randn(n,r),0); % random initialization
        %% Riemannian subgradient method
        % Riemannian subgradient method uses the SVD retraction mapping
        option_Rsub.F_manpg = -1e10;
        option_Rsub.phi_init = phi_init; option_Rsub.maxiter = n*r;  option_Rsub.tol = 5e-3;
        option_Rsub.r = r;    option_Rsub.n= n;  option_Rsub.mu=mu;  option_Rsub.type = 1;

        %     the solution of the Riemannian subgradient methods is
        %     used as the initial point for other algorithms
        [phi_init, F_Rsub,sparsity_Rsub,time_Rsub,...
            maxit_att_Rsub,succ_flag_sub]= Re_sub_grad(H,option_Rsub);

        x0 = phi_init;
        %% Drive_ManPG
        option.n = n; option.r = r; option.mu = mu;
        option.tol = 1e-8*n*r;
        option.maxiter = maxiteration;
        option.x0 = x0;
        option.stop = 1e-8;
        option.outputgap = outputgap;
        [x_manpg, fs_manpg, nv_manpg, iter_manpg,...
            sparsity_manpg, time_manpg, iter_time_manpg] = driver_ManPG(H, option, dx, V);
        tab(row + 1, 3, j) = iter_manpg;        tab(row + 1, 4, j) = fs_manpg(end);  
        tab(row + 1, 5, j) = nv_manpg(end);     tab(row + 1, 6, j) = time_manpg;    
        tab(row + 1, 7, j) = sparsity_manpg;


        %% Drive_ManPG_Ada
        [x_manpg_ada, fs_manpg_ada, nv_manpg_ada,...
            iter_manpg_ada, sparsity_manpg_ada, time_manpg_ada,iter_time_manpg_ada] = driver_ManPG_ada(H, option, dx, V);

        tab(row + 2, 3, j) = iter_manpg_ada;        tab(row + 2, 4, j) = fs_manpg_ada(end);  
        tab(row + 2, 5, j) = nv_manpg_ada(end);     tab(row + 2, 6, j) = time_manpg_ada;    
        tab(row + 2, 7, j) = sparsity_manpg_ada;

        %% ManPQN
        option.type = 0;
        option.inner_iter = 200;
        M = 5;
        [X_pqn, F_pqn,F_pqn_list,sp_pqn,t_pqn,...
            maxit_att_pqn,succ_flag_pqn,lins_pqn,in_av_pqn,nv_pqn, iter_time_pqn]= manpqn(H,option,dx,V,M,0);

        succ_no_pqn = 1;
        tab(row + 3, 3, j) = sum(maxit_att_pqn)/succ_no_pqn;       tab(row + 3, 4, j) = sum(F_pqn)/succ_no_pqn;  
        tab(row + 3, 5, j) = nv_pqn(end);                          tab(row + 3, 6, j) = sum(t_pqn)/succ_no_pqn;    
        tab(row + 3, 7, j) = sum(sp_pqn)/succ_no_pqn;


        %% Drive_RPN-CG
        [x_rpncg, fs_rpncg, nv_rpncg, iter_rpncg,...
            sparsity_rpncg, time_rpncg, iter_time_rpncg] = driver_rpncg(H, option, dx, V);
        tab(row + 4, 3, j) = iter_rpncg;        tab(row + 4, 4, j) = fs_rpncg(end);  
        tab(row + 4, 5, j) = nv_rpncg(end);     tab(row + 4, 6, j) = time_rpncg;    
        tab(row + 4, 7, j) = sparsity_rpncg;

        %% Drive_RPN-CGH
         option.epsilon = 1e-2;
        [x_rpncgh, fs_rpncgh, nv_rpncgh, iter_rpncgh,...
            sparsity_rpncgh, time_rpncgh, iter_time_rpncgh] = driver_rpn_cgh(H, option, dx, V);
        tab(row + 5, 3, j) = iter_rpncgh;        tab(row + 5, 4, j) = fs_rpncgh(end); 
        tab(row + 5, 5, j) = nv_rpncgh(end);     tab(row + 5, 6, j) = time_rpncgh;
        tab(row + 5, 7, j) = sparsity_rpncgh;        
         
    end

end

avetab = mean(tab, 3);
avetab(:,2) =[1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5]; % 1: ManPG, 2: ManPG-Ada, 3: ManPQN, 4: RPN-CG, 5:RPN-CGH

fout = fopen('cm_table_mu.txt','w');
for i = 1 : size(avetab, 1)
    for j = 1 : size(avetab, 2)
        if(j == 1 )
           fprintf(fout, '%1.2f & ', avetab(i, j)); 
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


