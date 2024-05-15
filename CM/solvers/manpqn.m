function [X_nls, F_nls,F,sparsity,time_nls,iter,flag_succ,num_linesearch,mean_ssn, normv, iter_time] =manpqn(H,option,d_l,V,M,bb_mod)
% solving min -Tr(X'*H*X)+ mu*norm(X,1) s.t. X'*X=Ir.
% parameters:
%   option.phi_init: initial iterative point
%   option.maxiter: maximal ietration number
%   option.tol: tolerance for stopping criterion
%   option.r, option.n, option.mu
%   option.inner_iter: maximal iteration number for SSN methods in the
%   inner loop
%   M£ºparameter for nonmonotone line search
%   bb_mod: the way computing BB stepsize strategy
%   retrac_mod: the way of choosing retraction
%%
%parameters
%tic;
retrac_mod = 1;
r = option.r;  % number of col
n = option.n;  % dim
mu = option.mu;
maxiter =option.maxiter;
tol = option.tol;
stop_label = option.stop;
%inner_tol = option.inner_tol;
inner_iter = option.inner_iter;
h = @(X) sum(mu.*sum(abs(X)));  % norm(X,1)
prox_fun = @(b,lambda,r) proximal_l1(b,lambda,r);
prox_fun_diag = @(b,lambda,r,h) proximal_l1_diag(b,lambda,r,h);
inner_flag = 0;
%setduplicat_pduplicat(r);
Dn = sparse(DuplicationM(r));
pDn = (Dn'*Dn)\Dn';
t_min=1e-4; % minimum stepsize
normv = [];
%%
%initial point
X = option.x0;
L = 8/d_l^2.*(sin(pi/4))^2 + V;
%dx = option.L/n;
%LAM_A =  (cos(2*pi*[0:n-1]'/n)-1)/dx^2;
%HX = real(fft(bsxfun(@times,ifft( X ),LAM_A)));
HX = H*X;

F(1) = -sum(sum(X.*(HX)))+h(X);
num_inner = zeros(maxiter,1);
opt_sub = num_inner;
num_linesearch = 0;
alpha = 1;
t0 = 1/L; t =t0;
hk = ones(n, 1);
para_step = 1.1;
linesearch_flag = 0;
pg_flag = 0;
% ls_flag_sum = 0;
num_inexact = 0;
innertol = max(1e-13, min(1e-11,1e-3*sqrt(tol)*t^2));
Cval = F(1);
nv = 1;
tic;
for iter = 2:maxiter
    
    %   fprintf('manpg__________________iter: %3d, \n', iter);
    ngx = HX; % negative gradient 2AX
    %xgx = X'*ngx;
    %pgx = ngx - 0.5*X*(xgx+xgx');
    pgx= ngx; % grad or projected gradient both okay
    %% subproblem
%     if alpha < t_min || num_inexact > 10
%         inner_tol = max(5e-16, min(1e-14,1e-5*tol*t^2)); % subproblem inexact;
%     else
%         inner_tol = max(1e-13, min(1e-11,1e-3*tol*t^2));
%     end

    innertol = min(max(1e-30, nv * nv * 1e-8), innertol);
    
    if iter == 3
        HXk_1 = H*Xk_1;
        gk_1 = -2*HXk_1 + 2*(Xk_1*(Xk_1'*HXk_1));
        gk = -2*HX + 2*(X*(X'*HX));
    elseif iter > 3
        gk_1 = gk;
        gk = -2*HX + 2*(X*(X'*HX));
        sk_1 = X-Xk_1;
        yk_1 = gk-gk_1;
        rho_k = abs(sum(sum(sk_1.*yk_1)));
        if bb_mod == 2
            if mod(iter,2) == 1
                t_bb = norm(sk_1, 'fro')/rho_k;
            else
                t_bb = rho_k/norm(yk_1, 'fro');
            end
        elseif bb_mod == 1
            t_bb = (norm(sk_1, 'fro')) / rho_k;
        else 
            t_bb = rho_k / (norm(yk_1, 'fro'));
        end
        t = max(t0, t_bb);
    end
    if linesearch_flag == 0
        t = t*para_step;
    else
        t = max(t0,t/para_step);
%         ls_flag_sum = ls_flag_sum + 1;
    end
    linesearch_flag = 0;
    
    if iter == 4
        h0 =  rho_k / (norm(yk_1, 'fro'));
        rho_k = 1/rho_k;
        Hk = h0 * eye(n) + (rho_k-1/rho_k) * (yk_1*yk_1');
        hk = diag(Hk);
    elseif iter > 4
        rho_k = 1/rho_k;
        vk = yk_1'*Hk;
        Hk = Hk - (vk'*vk)/abs(trace(yk_1'*Hk*yk_1)) + rho_k*(sk_1*sk_1');
        hk = diag(Hk);
    end

    hk = abs(hk);
    hk(hk<1e-4)=1e-4;
    invDiag = hk;

    if iter == 2
        [ PY, num_inner(iter), Lam, opt_sub(iter), ...
            in_flag]=Semi_newton_matrix_prox_diag(n,r,X,invDiag,t,...
            X+2*t*invDiag.*pgx,mu*t,innertol,prox_fun_diag,inner_iter,zeros(r),Dn,pDn);
    else
        [ PY, num_inner(iter), Lam, opt_sub(iter), ...
            in_flag]=Semi_newton_matrix_prox_diag(n,r,X,invDiag,t,...
            X+2*t*invDiag.*pgx,mu*t,innertol,prox_fun_diag,inner_iter,Lam,Dn,pDn);
    end

    if in_flag == 1   % subprolem total iteration.
        inner_flag = 1 + inner_flag;
    end
    alpha=1;
    D = PY-X; %descent direction D
    
    if retrac_mod == 1  % use SVD retraction mapping
        [U, SIGMA, S] = svd(PY'*PY);
        SIGMA =diag(SIGMA);
        Z = PY*(U*diag(sqrt(1./SIGMA))*S');
        % [Z,R]=qr(PY,0);       Z = Z*diag(sign(diag(R))); %old version need consider the sign
    elseif retrac_mod == 0  % use polar decomposition retraction mapping
        Z = PY * (eye(r)+D'*D)^(-1/2);
    end
        
    HZ = H*Z;
    % HZ = real(fft(bsxfun(@times,ifft( Z ),LAM_A)));
    % AZ = real(ifft( LAM_manpg.*fft(Z) ));
    
    F_trial = -sum(sum(Z.*(HZ)))+h(Z);
    nv = norm(D,'fro');
    normDsquared = nv^2;
    
%     if  normDsquared/t^2 < tol
%     if  trace(D'*(invDiag.*D)/(t^2)) < tol
%         % if  abs(F(iter)-F(iter-1))/(abs(F(iter))+1)<tol
%         break;
%     end

    if(mod(iter, option.outputgap) == 0)
        fprintf('iter:%d, Fval:%e, nv:%e,  num_linesearch:%d\n', iter, F_trial, nv, num_linesearch);
    end
    normv(iter-1) = nv;
    iter_time(iter-1) = toc;
    if nv < stop_label
        break;
    end
    
    %% nonmonotone-linesearch
    if iter <= M+1
        Cval = max(F(1:iter-1));
    else
        Cval = max(F(iter-M:iter-1));
    end
    while F_trial>= Cval-0.5/t*alpha*normDsquared
        alpha = 0.5*alpha;
        linesearch_flag = 1;
        num_linesearch = num_linesearch+1;
        if alpha<t_min
            num_inexact = num_inexact + 1;
            break;
        end
%         if alpha< 1e-16
%             break;
%         end
        PY = X + alpha*D;
        %  [U,~,V]=svd(PY,0);  Z=U*V';
        %  [Z,R]=qr(PY,0);   Z = Z*diag(sign(diag(R)));  %old version need consider the sign
        if retrac_mod == 1  % use SVD retraction mapping
            [U, SIGMA, S] = svd(PY'*PY);
            SIGMA =diag(SIGMA);
            Z = PY*(U*diag(sqrt(1./SIGMA))*S');
        elseif retrac_mod == 0  % use polar decomposition retraction mapping
            Z = PY * (eye(r)+D'*D)^(-1/2);
        end
        
%         [U, SIGMA, S] = svd(PY'*PY);
%         SIGMA =diag(SIGMA);
%         Z = PY*(U*diag(sqrt(1./SIGMA))*S');
        HZ= H*Z;
        %HZ = real(fft(bsxfun(@times,ifft( Z ),LAM_A)));
        F_trial =  -sum(sum(Z.*(HZ))) + h(Z);
    end
    Xk_1 = X;
    X = Z;
    HX = HZ;
    F(iter) = F_trial;

    
%     if  pg_flag == 0 && normDsquared/t^2 < min(tol, 1e-6)% && F_trial < option.F_manpg+1e-4
% %         break;
%         pg_flag = 1;
%     end
%     if  pg_flag == 1 && F_trial < option.F_manpg+1e-3
%         break;
%     end
end

X((abs(X)<=1e-5))=0;
X_nls=X;
time_nls = toc;
mean_ssn = sum(num_inner)/(iter-1);

% if iter == maxiter && sqrt(normDsquared)/t > 1e-1
%     flag_succ = 0;
%     sparsity  = 0;
%     F_nls = 0;
%     time_nls = 0;
% else
    flag_succ = 1;
    sparsity= sum(sum(X_nls==0))/(n*r);
    F_nls =  F(iter-1);
    
%     fprintf('NLS:Iter ***  Fval *** CPU  **** sparsity ***inner_inexact&averge_No. ** opt_norm ** total_linsea \n');
%     print_format = ' %i     %1.5e    %1.2f     %1.2f         %4i   %2.2f                %1.3e        %d \n';
%     fprintf(1,print_format, iter-1,min(F), time_nls,sparsity, inner_flag, sum(num_inner)/(iter-1) ,sqrt(normDsquared)/t,num_linesearch);
    fprintf('ManPQN:Iter ***  Fval *** CPU  **** sparsity ***inner_inexact&averge_No. ** opt_norm ** total_linsea \n');
    print_format = ' %i     %1.5e    %1.2f     %1.2f         %4i   %2.2f                %1.3e        %d \n';
    fprintf(1,print_format, iter-1,min(F_nls), time_nls,sparsity, inner_flag, mean_ssn ,sqrt(normDsquared)/t,num_linesearch);
end

