clear;clc
close all;


r = 5; % r number of column
mu = .8; % mu sparse parameter

m = 40; n = 4000;
noiselevel = 0.8;% 0.5;
t = linspace(0, 1, n);

pc1 = max(0, ((t - 0.6)> 0) & (0.8 - t > 0))*0.5;
pc2 = max(0, ((t - 0.2)> 0) & (0.4 - t > 0))*0.5;
pc3 = 0.8*exp(-(t - 0.5).^2/5e-3);
pc4 = 0.4*exp(-(t - 0.15).^2/1e-3) + 0.4*exp(-(t - 0.85).^2/1e-3);
pc5 = 0.4*exp(-(t - 0.05).^2/1e-3) - 0.4*exp(-(t - 0.95).^2/1e-3);
PC = [pc1; pc2; pc3; pc4; pc5];

truesparsity = sum(sum(PC~=0)) / length(PC(:));
A = [ones(m/5,1)*pc1 + randn(m/5,n) * noiselevel; ...
    ones(m/5,1)*pc2 + randn(m/5,n) * noiselevel; ...
    ones(m/5,1)*pc3 + randn(m/5,n) * noiselevel; ...
    ones(m/5,1)*pc4 + randn(m/5,n) * noiselevel; ...
    ones(m/5,1)*pc5 + randn(m/5,n) * noiselevel];
tmp = sqrt(sum(A .* A));
A = A ./ repmat(tmp, m, 1);


seed = round(rand() * 10000000);
seed = 1;
rng(seed);


% random initialization
% x0 = randn(n,r);
% x0 = x0/norm(x0,'fro');
[U, S, V] = svd(A', 0);
x0 = U(:, 1 : r);

fid = 1;
maxiteration = 5000;
outputgap = 100;


%% Drive_ManPG
option.n = n; option.r = r; option.mu = mu;
option.tol = 1e-8*n*r;
option.maxiter = maxiteration;
option.x0 = x0;
option.stop = 1e-10;
option.outputgap = outputgap;

[x_manpg, fs_manpg, nv_manpg, iter_manpg,...
    sparsity_manpg, time_manpg,iter_time_manpg] = driver_ManPG(A, option);
Fs_manpg = fs_manpg(end);
Ns_manpg = nv_manpg(end);

%% Drive_ManPG_Ada
[x_manpg_ada, fs_manpg_ada, nv_manpg_ada,...
    iter_manpg_ada, sparsity_manpg_ada, time_manpg_ada,iter_time_manpg_ada] = driver_ManPG_ada(A, option);
Fs_manpg_ada = fs_manpg_ada(end);
Ns_manpg_ada = nv_manpg_ada(end);


%% ManPQN
option.type =0;
option.inner_iter = 200;
M = 5;
[X_pqn, F_pqn,F_pqn_list,sp_pqn,t_pqn,...
    maxit_att_pqn,succ_flag_pqn,lins_pqn,in_av_pqn,nv_pqn, iter_time_pqn]= manpqn_orth_sparse(A,option,M,1);
succ_no_pqn = 1;
iter_pqn = sum(maxit_att_pqn)/succ_no_pqn;
time_pqn = sum(t_pqn)/succ_no_pqn;
Fs_pqn   = sum(F_pqn)/succ_no_pqn;
sparsity_pqn   = sum(sp_pqn)/succ_no_pqn;
Ns_pqn   = nv_pqn(end);


%% Drive_RPN-CG
option.epsilon = 1e-3;
[x_rpncg, fs_rpncg, nv_rpncg, iter_rpncg,...
    sparsity_rpncg, time_rpncg, iter_time_rpncg] = driver_rpncg(A, option);
Fs_rpncg = fs_rpncg(end);
Ns_rpncg = nv_rpncg(end);

%% Drive_RPN-CGH
option.epsilon = 1e-3;
[x_rpncgh, fs_rpncgh, nv_rpncgh, iter_rpncgh, sparsity_rpncgh, time_rpncgh, iter_time_rpncgh] = driver_rpn_cgh(A, option);
Fs_rpncgh = fs_rpncgh(end);
Ns_rpncgh = nv_rpncgh(end);

fprintf(fid,' Alg ****        Iter *****  Fval ******* sparsity ***** cpu ******normv**\n');
print_format =  'ManPG     : &  %e   & %1.5e  &   %1.2f    &   %3.2f   & %1.3e\n';
fprintf(fid, print_format, iter_manpg , Fs_manpg, sparsity_manpg, time_manpg, Ns_manpg);
print_format =  'ManPG-Ada : &  %e   & %1.5e  &   %1.2f    &   %3.2f   & %1.3e\n';
fprintf(fid, print_format, iter_manpg_ada , Fs_manpg_ada, sparsity_manpg_ada, time_manpg_ada, Ns_manpg_ada);
print_format =  'ManPQN    : &  %e   & %1.5e  &   %1.2f    &   %3.2f   & %1.3e\n';
fprintf(fid,print_format, iter_pqn , Fs_pqn, sparsity_pqn, time_pqn, Ns_pqn);
print_format =  'RPN-CG    : &  %e   & %1.5e  &   %1.2f    &   %3.2f   & %1.3e \n';
fprintf(fid, print_format, iter_rpncg,  Fs_rpncg, sparsity_rpncg, time_rpncg, Ns_rpncg);
print_format =  'RPN-CGH   : &  %e   & %1.5e  &   %1.2f    &   %3.2f   & %1.3e \n';
fprintf(fid, print_format, iter_rpncgh, Fs_rpncgh, sparsity_rpncgh, time_rpncgh, Ns_rpncgh);

clf;
set(0,'defaultaxesfontsize',15, ...
   'defaultaxeslinewidth',0.7, ...
   'defaultlinelinewidth',.8,'defaultpatchlinewidth',0.8);
set(0,'defaultlinemarkersize',10)

subplot(1,2,1)

semilogy(1:length(nv_manpg), nv_manpg,'r.-','LineWidth',1.3)
hold on
semilogy(1:length(nv_manpg_ada), nv_manpg_ada,'-','LineWidth',1.3)
hold on
semilogy(1:length(nv_pqn), nv_pqn,'m--', 'LineWidth',1.3)
hold on
semilogy(1:length(nv_rpncg), nv_rpncg,'g-+', 'LineWidth',1.3)
hold on
semilogy(1:length(nv_rpncgh), nv_rpncgh,'b-o', 'LineWidth',1.3)
legend('ManPG','ManPG-Ada','ManPQN','RPN-CG','RPN-CGH')
xlabel('Iter');
ylabel('$\|v(x_k)\|$','Interpreter','latex')
title(' Synthetic data ')
 
subplot(1,2,2)

semilogy(iter_time_manpg, nv_manpg,'r.-','LineWidth',1.3)
hold on
semilogy(iter_time_manpg_ada, nv_manpg_ada,'-','LineWidth',1.3)
hold on
semilogy(iter_time_pqn, nv_pqn,'m--', 'LineWidth',1.3)
hold on
semilogy(iter_time_rpncg, nv_rpncg,'g-+', 'LineWidth',1.3)
hold on
semilogy(iter_time_rpncgh, nv_rpncgh,'b-o', 'LineWidth',1.3)
legend('ManPG','ManPG-Ada','ManPQN','RPN-CG','RPN-CGH')
xlabel('CPU time(s)');
ylabel('$\|v(x_k)\|$','Interpreter','latex')
title(' Synthetic data ')


