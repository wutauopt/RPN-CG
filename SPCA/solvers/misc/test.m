%function compare_spca

clc
clear;
close all;
% addpath ../misc
% addpath ../SSN_subproblem

% profile on

n_set = [100];  % [100;200;500;800;1000;1500]; %dimension
r_set = [5]; % [1;2;4;6;8;10];   % rank

mu_set = [.7];  % [0.55;0.6;0.65;0.7;0.75;0.8];
% M_set = [5;10;20;30];
index = 1;
M = 5;
m_lbfgs = 3;
exp_time = 1;

id_n = 1:length(n_set);   
n = n_set(id_n);
fid =1;

id_r = 1:length(r_set);  % r  number of column
r = r_set(id_r);
id_mu = 1:length(mu_set); % mu  sparse parameter
mu = mu_set(id_mu);

succ_no_manpg = 0;  
succ_no_pn = 0;  
            
fprintf(fid,'==============================================================================================\n');
fprintf(fid,'- n -- r -- mu --------\n');
fprintf(fid,'%4d %3d  %3.2f \n',n,r,mu);

% for test_random = 1:50  %times average.
    rng('shuffle');
    m = 50;
    B = randn(m,n);
    type = 0; % random data matrix
    if (type == 1) %covariance matrix
        scale = max(diag(B)); % Sigma=A/scale;
    elseif (type == 0) %data matrix
        B = B - repmat(mean(B,1),m,1);
        B = normc(B);
    end
    rng('shuffle');
    [phi_init,~] = svd(randn(n,r),0);  % random intialization
    %[phi_init,~] = eigs(H,r);    % singular value initialization
%     option_Rsub.F_manpg = -1e10;
%     option_Rsub.phi_init = phi_init; option_Rsub.maxiter = 5e2;  option_Rsub.tol = 5e-3;
%     option_Rsub.r = r;    option_Rsub.n= n;  option_Rsub.mu=mu;  option_Rsub.type = type;
% 
%     [phi_init, F_Rsub(test_random),sparsity_Rsub(test_random),time_Rsub(test_random),...
%         maxit_att_Rsub(test_random),succ_flag_sub]= Re_sub_grad_spca(B,option_Rsub);


    %%  manpg parameter
    option_manpg.adap = 0;    option_manpg.type =type;
    option_manpg.phi_init = phi_init; option_manpg.maxiter = 3000;  %option_manpg.tol =1e-8*n*r;
    option_manpg.tol =1e-5;
    option_manpg.r = r;    option_manpg.n = n;  option_manpg.mu = mu;
    option_manpg.inner_iter = 100;

    %% ManPG
    [X_manpg, F_manpg,F_manpg_list, sparsity_manpg,time_manpg,...
        maxit_att_manpg,succ_flag_manpg, lins,in_av,nv_manpg]= manpg_orth_sparse(B,option_manpg);
    if succ_flag_manpg == 1
        succ_no_manpg = succ_no_manpg + 1;
    end

    %option_manpg.F_manpg = F_manpg;
    %%  ManPQN algorithm with proximal newton method (approximated by diagonal matrix)
    [X_pn, F_pn,F_pn_list,sparsity_pn,time_pn,maxit_att_pn,succ_flag_pn,lins_pn,in_av_pn,nv_pn]= manpqn_orth_sparse(B,option_manpg,M,1);
    if succ_flag_pn == 1
        succ_no_pn = succ_no_pn + 1;
    end


iter.manpg = sum(maxit_att_manpg)/succ_no_manpg;
time.manpg = sum(time_manpg)/succ_no_manpg;
Fval.manpg = sum(F_manpg)/succ_no_manpg;
Sp.manpg = sum(sparsity_manpg)/succ_no_manpg;


iter.pn = sum(maxit_att_pn)/succ_no_pn;
time.pn = sum(time_pn)/succ_no_pn;
Fval.pn = sum(F_pn)/succ_no_pn;
Sp.pn   = sum(sparsity_pn)/succ_no_pn;

fprintf(fid,' Alg ****        Iter *****  Fval ******* sparsity ***** cpu ******normv**\n');

print_format =  'ManPG       &   %.2f   & %1.5e  &    %1.2f  &   %3.4f  & %1.3e  \\\\ \n';
fprintf(fid, print_format, iter.manpg, Fval.manpg, Sp.manpg,time.manpg,nv_manpg(end));
print_format =  'ManPQN      &   %.2f   & %1.5e  &    %1.2f  &   %3.4f  & %1.3e  \\\\ \n';
fprintf(fid, print_format, iter.pn, Fval.pn, Sp.pn,time.pn, nv_pn(end));









