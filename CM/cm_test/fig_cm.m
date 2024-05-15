clc
clear;
close all;

seed = round(rand() * 10000000);
seed = 10;
fprintf('seed:%d\n', seed);
rng(seed);

n = 256; 
r = 8;
mu = 0.1;


fid = 1;
L = 50; dx = L/n;  V = 0;
H = -Sch_matrix(0,L,n);
Hmaxsigma = svds(H, 1);
% H = speye(n) * Hmaxsigma + H;

maxiter = 3000;
outputgap = 100;



[phi_init,~] = svd(randn(n,r),0);  % randomly generate initial point

%% Riemannian subgradient method
% Riemannian subgradient method uses the SVD retraction mapping
option_Rsub.F_manpg = -1e10;
option_Rsub.phi_init = phi_init; option_Rsub.maxiter = n*r;  option_Rsub.tol = 5e-3;
option_Rsub.r = r;    option_Rsub.n= n;  option_Rsub.mu=mu;  option_Rsub.type = 1;

%     the solution of the Riemannian subgradient methods is
%     used as the initial point for other algorithms
[phi_init, F_Rsub,sparsity_Rsub,time_Rsub,...
    maxit_att_Rsub,succ_flag_sub]= Re_sub_grad(H,option_Rsub);


%% Drive_ManPG
option.n = n; option.r = r; option.mu = mu;
option.tol = 1e-8*n*r;
option.inner_iter = 200;
option.maxiter = maxiter;
option.x0 = phi_init;
option.stop = 1e-8;
option.outputgap = outputgap;

[x_manpg, fs_manpg, nv_manpg, iter_manpg,...
    sparsity_manpg, time_manpg,iter_time_manpg] = driver_ManPG(H, option, dx, V);
Fs_manpg = fs_manpg(end);
Ns_manpg = nv_manpg(end);

%% Drive_ManPG_Ada
[x_manpg_ada, fs_manpg_ada, nv_manpg_ada, iter_manpg_ada,...
    sparsity_manpg_ada, time_manpg_ada,iter_time_manpg_ada] = driver_ManPG_ada(H, option, dx, V);
Fs_manpg_ada = fs_manpg_ada(end);
Ns_manpg_ada = nv_manpg_ada(end);

%%  ManPG-NLS algorithm with proximal newton method (approximated by diagonal matrix)
%option.maxiter = maxiteration;
M = 5;  % nonmonotone line search parameter
[X_pqn, F_pqn,F_pqn_list, sp_pqn,t_pqn,...
    maxit_att_pqn,succ_flag_pqn,lins_pqn,in_av_pqn,nv_pqn,iter_time_pqn]= manpqn(H,option,dx,V,M,0);
succ_no_pqn = 1;
iter_pqn = sum(maxit_att_pqn)/succ_no_pqn;
time_pqn = sum(t_pqn)/succ_no_pqn;
Fs_pqn   = sum(F_pqn)/succ_no_pqn;
sparsity_pqn   = sum(sp_pqn)/succ_no_pqn;
Ns_pqn   = nv_pqn(end);


%% Drive_RPN-CG
[x_rpncg, fs_rpncg, nv_rpncg, iter_rpncg,...
    sparsity_rpncg, time_rpncg, iter_time_rpncg] = driver_rpncg(H, option, dx,V);
Fs_rpncg = fs_rpncg(end);
Ns_rpncg = nv_rpncg(end);

%% Drive_RPN-CGH
option.epsilon = 1e-2;
[x_rpncgh, fs_rpncgh, nv_rpncgh, iter_rpncgh,...
    sparsity_rpncgh, time_rpncgh, iter_time_rpncgh] = driver_rpn_cgh(H, option, dx,V);
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
title(' n = 256 , r = 8, \mu = 0.15 ')
 
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
title(' n = 256 , r = 8, \mu = 0.15 ')
