function [w, inner_it, status] = tCG(BB,v,nv,v_bar,v_hat,x,gf,id1,id2,t,tau,mu,params)
% min l_x^{T}w + .5*w^{T}BB_1*w s.t. Bx_bar^{T}w = 0
% Parameters
% sigma > 0, gamma>0, tau > 0, theta > 0, kappa \in [0,1]
%
% OUTPUT:
% eta : the final iterate
% inner_it : the number of iterations in TCG
% status : the status of the output in TCG
%     -1: attain maximum iteration
%      0：'early 1'
%      1: 'early 2'
%      2: 'neg'
%      3: 'early 3'
%      4: 'lin'
%      5: 'sup' 

status = -1; % maximum iteration
inner_it = 0;
[n,m] = size(v);
z = BB(v);
tau_nv_hat_2 = tau * (v_hat(:)'*v_hat(:));
v_Bv = v(:)'*z(:);

tol1 = v_Bv + tau_nv_hat_2;
tol2 = trace(gf'*v) + .5*tol1 + mu * sum(abs(x.x_v(:))) - x.hx; % G_x(v) - G_x(0)
if tol2 > 0 % G_x(v) > G_x(0)
    w = zeros(size(v_bar));
    status = 0;
    return;
end
 
if tol1  < params.gamma*nv^2
    w = zeros(size(v_bar));
    status = 1;
    return;
end

d = .5*m*(m+1);
r2 = sqrt(2);
tmpp = zeros(m, m, d);
idx = 0;
for i = 1 : m
    idx = idx + 1;
    tmpp(i, i, idx) = 1;
end
for i = 1 : m
    for j = i + 1 : m
        idx = idx + 1;
        tmpp(i, j, idx) = 1 / r2;
        tmpp(j, i, idx) = tmpp(i, j, idx);
    end
end
tmpp = sparse(reshape(tmpp, m, m * d));
Bvtmp = reshape(full(x.main * tmpp), n * m, d);
Bbar = Bvtmp(id1, :);
D = Bbar'*Bbar; % Dinv = (\bar_{B}^{\T}\bar_{B})^{\dagger}
[U, S, V] = svd(D);
Sinv = zeros(d, d);
Sinvdiag = 1./S(1:d+1:end);
Sinvdiag(find(Sinvdiag > 1e8)) = 0;
Sinv(1:d+1:end) = Sinvdiag;
Dinv = V * Sinv * U';

P_x = @(eta)Pxx(x,eta,Bbar,Dinv);
l_x = -1/t *v_bar + z(id1); % BB11(v_bar) + BB12(v_hat);

w = zeros(size(v_bar));
r = P_x(l_x); % r0 = P_x(l_x)

o = -r;
delta = r(:)'*r(:);
r_r = delta;
norm_r0 = norm(r,'fro');
tt = z;

for j = 1:params.max_innit

    VV = zeros(n, m);
    VV(id1) = o;
    ppp = BB(VV);
    
    p = ppp(id1); % q = P_x(p);
    o_q = o(:)'*p(:);  % o_q = o_p;
    if(o_q <= params.vartheta*delta)
        status = 2; % negative curture
        break;
    end

    % compute the step size alpha
    alpha = r_r/o_q;
    w_old = w;
    w = w + alpha*o;

    % update the residual
    q = P_x(p);
    r = r + alpha * q;
    d = zeros(n,m);
    d(id1) = v_bar + w;
    d(id2) = v_hat;

    tt = tt + alpha * ppp;
    t = tt;
    
    err_1 = d(:)'*t(:) + tau_nv_hat_2;
    x_d = x.main + d;
    err_2 = gf(:)'*d(:) + .5*err_1 + mu * sum(abs(x_d(:))) - x.hx; % G_x(d) - G_x(0)
   
   if( err_1 <  params.gamma* (d(:)'*d(:)) || err_2 > 0)
       w = w_old;
       status = 3;
       break;
   end

%    if (err_2 > 0)
%         w = w_old;
%         status = 3;
%         break;
%    end


    rold_rold = r_r;
    r_r = r(:)'*r(:);
    norm_r = norm(r,'fro');
    
    % err_1- params.gamma*norm(d,'fro')^2 >=0 
    % err_2 <=0
    % fprintf('innit:%d, err_1:%e, err_2:%e, nr0:%e, nr:%e\n', j,err_1-params.gamma*norm(d,'fro')^2,err_2,norm_r0,norm_r);

     % check residual stopping criterion ( kappa/theta)
      if(norm_r <= norm_r0 * min([norm_r0^params.theta,params.kappa]))
            % residual is small enough to quit
            if(params.kappa < norm_r0^params.theta)
                % Stop due to kappa
                status = 4; % linear convergence
            else
                % Stop due to theta
                status = 5; % superlinear convergence
            end
            break;
       end

       % compute new search direction
       beta = r_r / rold_rold;
       o = -r + beta * o;
       delta = r_r + beta^2 * delta;
end
   inner_it = j;
   if status == -1 % attain maximun iteration
       w = zeros(size(v_bar));
   end
   return;
end

function [output] = Pxx(x,eta,Bbar,Dinv)
    output = eta - Bbar * (Dinv * (Bbar'*eta));
end


    


           




