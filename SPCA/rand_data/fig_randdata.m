clear;clc
close all;

n = 400; % n dimension
r = 12; % r number of column
mu = .8; % mu sparse parameter

seed = round(rand() * 10000000);
seed = 2;
fprintf('seed:%d\n', seed);
rng(seed);


fid = 1;  
maxiter = 5000;
outputgap = 100;

% generate the random data matrix A
m = 50;
A = randn(m,n);
A = A - repmat(mean(A,1),m,1);
A = normc(A);

% [phi_init, ~] = svd(randn(n,r),0); % random initialization
% x0 = phi_init;
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
    sparsity_manpg, time_manpg,iter_time_manpg] = driver_ManPG(A, option);
Fs_manpg = fs_manpg(end);
Ns_manpg = nv_manpg(end);

%% Drive_ManPG_Ada
[x_manpg_ada, fs_manpg_ada, nv_manpg_ada,...
    iter_manpg_ada, sparsity_manpg_ada, time_manpg_ada,iter_time_manpg_ada] = driver_ManPG_ada(A, option);
Fs_manpg_ada = fs_manpg_ada(end);
Ns_manpg_ada = nv_manpg_ada(end);

[U, S, V] = svd(x_manpg_ada' * x_manpg);
if norm(x_manpg_ada - x_manpg * V * U', 'fro') >= 1e-2
    error('ManPG-Ada: converge to different minimizers!\n');
end

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

[U, S, V] = svd(X_pqn' * x_manpg);
if norm(X_pqn - x_manpg * V * U', 'fro') >= 1e-2
    error('ManPQN: converge to different minimizers!\n');
end

%% Drive_RPN-CG
[x_rpncg, fs_rpncg, nv_rpncg, iter_rpncg, sparsity_rpncg, time_rpncg, iter_time_rpncg] = driver_rpncg(A, option);
Fs_rpncg = fs_rpncg(end);
Ns_rpncg = nv_rpncg(end);

[U, S, V] = svd(x_rpncg' * x_manpg);
if norm(x_rpncg - x_manpg * V * U', 'fro') >= 1e-2
    error('RPN-CG: converge to different minimizers!\n');
end

%% Drive_RPN-CGH
option.epsilon = 1e-2;
[x_rpncgh, fs_rpncgh, nv_rpncgh, iter_rpncgh, sparsity_rpncgh, time_rpncgh, iter_time_rpncgh] = driver_rpn_cgh(A, option);
Fs_rpncgh = fs_rpncgh(end);
Ns_rpncgh = nv_rpncgh(end);

[U, S, V] = svd(x_rpncgh' * x_manpg);
if norm(x_rpncgh - x_manpg * V * U', 'fro') >= 1e-2
    error('RPN-CGH: converge to different minimizers!\n');
end

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
title(' n = 400 , r = 12, \mu = 0.8 ')
 
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
title(' n = 400 , r = 12, \mu = 0.8 ')
