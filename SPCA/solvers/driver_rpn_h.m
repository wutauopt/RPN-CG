
function [xopt, fs, Ds, iter, sparsity, comtime, iter_time, status] = driver_rpn_h(A, option)
    % parameters
    n = option.n;
    r = option.r;
    option.d = r*(r+1)/2;
    option.k = n*r;
    mu = option.mu;
    tol = option.tol;
    maxiter = option.maxiter;
    stop_label = option.stop;
    epsilon = option.epsilon;
    x0 = option.x0;


    % parameters for line search
    delta = 0.001;
    
    % functions for the optimization problem
    fhandle = @(x)f(x, A, mu);
    gfhandle = @(x)gf(x, A, mu);
    Bv = @(x,v)embedded_full_normal_c(x, v);
    Bt_v = @(x,eta)embedded_intr_normal_c(x, eta);
    fprox = @prox;
    fcalJ = @calJ;
    
    % functions for the manifold
    fcalA = @calA;
    fcalAstar = @calAstar;
    
    xinitial.main = x0;
    L = 2 * norm(A)^2;
    t = 1/L;
    tic
    [x1, gf1, fs1, Ds1, iter1,iter1_time] = solver(fhandle, gfhandle, fcalA, fcalAstar, fprox, fcalJ, xinitial, t, tol, delta, maxiter, mu, epsilon, option);
    if iter1 < maxiter
       [xopt, fs2, Ds2, iter2, iter2_time,status] = solver1(A,x1,gf1,Ds1(end),fhandle, gfhandle, Bv, Bt_v, fcalA, fcalAstar, fprox, fcalJ, t, stop_label, iter1, option);
    else
        xopt = x1; fs2 = []; Ds2 = []; iter2 = 0; iter2_time = [];
    end
    comtime = toc;
    xopt.main(abs(xopt.main) < 1e-5) = 0;
    sparsity = sum(sum(abs(xopt.main) < 1e-5)) / (n * r);
    xopt = xopt.main;
    iter = iter1 + iter2;
    fs=[fs1';fs2'];
    Ds = [Ds1';Ds2'];
    iter_time = [iter1_time'; iter2_time'];
    fprintf('RPN-H:*** Iter ***  Fval *** CPU  **** sparsity *** opt_norm  \n');
    print_format = '     %i     %1.5e    %1.2f        %1.2f        %1.3e       \n';
    fprintf(1,print_format, iter,fs(end), comtime,sparsity,Ds(end));
end

function [xopt, gf2, fs, Ds, iter, iter_time] = solver(fhandle, gfhandle, fcalA, fcalAstar, fprox, fcalJ, x0, t, tol, delta, maxiter, mu, epsilon, option)
    err = inf;
    x1 = x0;
    x2 = x1; fs = []; Ds = [];
    [f1, x1] = fhandle(x1);
    gf1 = gfhandle(x1);
    iter = 0;
    fs(iter + 1) = f1;
    d = option.d;
    Dinitial = zeros(d,1);
    totalbt = 0;
    innertol = max(1e-13, min(1e-11,1e-3*sqrt(tol)*t^2));
    nv = 1;
    while(err > epsilon  && iter < maxiter) 
        innertol = min(max(1e-30, nv * nv * 1e-8), innertol);
        [v, Dinitial, inneriter] = finddir(x1, gf1, t, fcalA, fcalAstar, fprox, fcalJ, mu, Dinitial, innertol);
        nv = norm(v, 'fro'); nvsquared = nv^2;
        alpha = 1;
        x2 = R(x1, alpha * v);
        [f2, x2] = fhandle(x2);
        btiter = 0;
        while(f2 > f1 - delta * alpha * nvsquared && btiter < 3)
            alpha = .5* alpha;
            x2 = R(x1, alpha * v);
            [f2, x2] = fhandle(x2);
            btiter = btiter + 1;
            totalbt = totalbt + 1;
        end
        gf2 = gfhandle(x2);
        
        iter = iter + 1;
        err = nv;
        Ds(iter) = nv; 
        fs(iter + 1) = f2;
        iter_time(iter) = toc;
        if(mod(iter, option.outputgap) == 0)
            fprintf('iter:%d, f:%e, nv:%e, btiter:%d \n', iter, f1, nv, btiter);
        end
        x1 = x2; f1 = f2; gf1 = gf2;
    end
    fprintf('iter:%d, f:%e, nv:%e, ngf:%e, totalbt:%d\n', iter, f1, nv, norm(gf1, 'fro'), totalbt);
    xopt = x2;
end

function [xopt, fs, Ds, iter, iter_time, status] = solver1(A, x1, gf1, nv, fhandle,gfhandle, Bv, Bt_v, fcalA, fcalAstar, fprox, fcalJ, t, stoplabel,initer, option)
       num_u = 0;
       iter = 0;
       n = option.n;
       r = option.r;
       mu = option.mu;
       d = option.d;
       k = option.k;
       Dinitial = zeros(d,1);
       innertol = 1;
       status = 0; 
       while(nv > stoplabel  && iter + initer <  option.maxiter)
           innertol = min(max(1e-30, nv * nv * nv), innertol);
           [v, Dinitial, inneriter] = finddir(x1, gf1, t, fcalA, fcalAstar, fprox, fcalJ, mu, Dinitial, innertol);
           lambda = Dinitial;
           nv = norm(v,'fro');

           Blambda = Bv(x1,lambda);
           M = abs(x1.main - t*gf1 - t*Blambda) > t*mu;
           M1 = M;
           M = diag(M(:));
           % construct B_x
           I = eye(d);
           B = zeros(k,d);
           for i = 1:d
               w = I(:,i);
               Bv1 = Bv(x1, w);
               B(:,i) = Bv1(:);
           end 
           M_B = M * B; H = M_B'*M_B;
           Ju = @(z) Jx_u(x1, z, M1, Bt_v, Bv, Blambda, H, t, A);
           u = cgs(Ju,v(:),1e-5,1000);
           D = reshape(u,[n,r]);
           D = D - Bv(x1,Bt_v(x1,D));

           x2 = R(x1,D);
           [f2, x2] = fhandle(x2);
           num_u = num_u + 1;
           if num_u >= 20
               status = 1;
               break;
           end
           gf2 = gfhandle(x2);
           x1 = x2; f1 = f2; gf1 = gf2;
           iter = iter + 1;
           Ds(iter) = nv;
           fs(iter) = f2;
           iter_time(iter) = toc;
           fprintf('iter:%d, f:%e, nv:%e, nu:%e, num_u:%d\n', iter, f2, nv,norm(u,'fro'), num_u);
           
       end
       fprintf('iter:%d, f:%e, nv:%e, nu: %e, num_u:%d\n', iter+initer, f1, nv, norm(u,'fro'), num_u);
       xopt = x2;
end


function output = R(x, eta)
    [Q,R] = qr(x.main + eta,0);
    [U,S,V] = svd(R);
    output.main = Q*(U*V');
end

function [output, x] = f(x, A, mu)
    x.Ax = A * x.main;
    tmp = norm(x.Ax, 'fro');
    output = - tmp * tmp + mu * sum(abs(x.main(:)));
end

function output = gf(x, A, mu)
    output = -2 * (A' * x.Ax);
end

% compute E(Lambda)
function ELambda = E(Lambda, BLambda, x, gfx, t, fcalA, fcalAstar, fprox, fcalJ, mmu)
    if(length(BLambda) == 0)
        BLambda = x - t * (gfx - fcalAstar(Lambda, x));
    end
    DLambda = fprox(BLambda, t, mmu) - x;
    ELambda = fcalA(DLambda, x);
end

% compute calG(Lambda)[d]
function GLambdad = GLd(Lambda, d, BLambda, Blocks, x, gfx, t, fcalA, fcalAstar, fprox, fcalJ, mmu)
        GLambdad = t * fcalA(fcalJ(BLambda, fcalAstar(d, x), t, mmu), x);
end

% Use semi-Newton to solve the subproblem and find the search direction
function [output, Lambda, inneriter] = finddir(xx, gfx, t, fcalA, fcalAstar, fprox, fcalJ, mmu, x0, innertol)
    x = xx.main;
    lambda = 0.2;
    nu = 0.99;
    tau = 0.1;
    eta1 = 0.2; eta2 = 0.75;
    gamma1 = 3; gamma2 = 5;
    alpha = 0.1;
    beta = 1 / alpha / 100;
    [n, p] = size(x);
    
    z = x0;
    BLambda = x - t * (gfx - fcalAstar(z, x));
    Fz = E(z, BLambda, x, gfx, t, fcalA, fcalAstar, fprox, fcalJ, mmu);
    
    nFz = norm(Fz, 'fro');
    nnls = 5;
    xi = zeros(nnls, 1);% for non-monotonic linesearch
    xi(nnls) = nFz;
    maxiter = 1000;
    times = 0;
    Blocks = cell(p, 1);
    while(nFz * nFz > innertol && times < maxiter) % while not converge, find d and update z
        mu = lambda * max(min(nFz, 0.1), 1e-11);
        Axhandle = @(d)GLd(z, d, BLambda, Blocks, x, gfx, t, fcalA, fcalAstar, fprox, fcalJ, mmu) + mu * d;
        [d, CGiter] = myCG(Axhandle, -Fz, tau, lambda * nFz, 30); % update d
        u = z + d;
        Fu = E(u, [], x, gfx, t, fcalA, fcalAstar, fprox, fcalJ, mmu); 
        nFu = norm(Fu, 'fro');
        
        if(nFu < nu * max(xi))
            z = u;
            Fz = Fu;
            nFz = nFu;
            xi(mod(times, nnls) + 1) = nFz;
            status = 'success';
        else
            rho = - sum(Fu(:) .* d(:)) / norm(d, 'fro')^2;
            if(rho >= eta1)
                v = z - sum(sum(Fu .* (z - u))) / nFu^2 * Fu;
                Fv = E(v, [], x, gfx, t, fcalA, fcalAstar, fprox, fcalJ, mmu);
                nFv = norm(Fv, 'fro');
                if(nFv <= nFz)
                    z = v;
                    Fz = Fv;
                    nFz = nFv;
                    status = 'safegard success projection';
                else
                    z = z - beta * Fz;
                    Fz = E(z, [], x, gfx, t, fcalA, fcalAstar, fprox, fcalJ, mmu);
                    nFz = norm(Fz, 'fro');
                    status = 'safegard success fixed-point';
                end
            else
%                 fprintf('unsuccessful step\n');
                status = 'safegard unsuccess';
            end
            if(rho >= eta2)
                lambda = max(lambda / 4, 1e-5);
            elseif(rho >= eta1)
                lambda = (1 + gamma1) / 2 * lambda;
            else
                lambda = (gamma1 + gamma2) / 2 * lambda;
            end
        end
        BLambda = x - t * (gfx - fcalAstar(z, x));
%         fprintf(['iter:%d, nFz:%f, xi:%f, ' status '\n'], times, nFz, max(xi));
        times = times + 1;
    end
    Lambda = z;
    inneriter = times;
    output = fprox(BLambda, t, mmu) - x;
end

function output = prox(X, t, mu)
    output = min(0, X + t * mu) + max(0, X - t * mu);
end

function output = calA(Z, U) % U \in St(p, n)
   x.main = U;
   output = embedded_intr_normal_c(x, Z);
   output = -output;
end

function output = calAstar(Lambda, U) % U \in St(p, n)
   x.main = U;
   output = embedded_full_normal_c(x, Lambda);
   output = -output;
end

function output = calJ(y, eta, t, mu)
    output = (abs(y) > mu * t) .* eta;
end

function [output, k] = myCG(Axhandle, b, tau, lambdanFz, maxiter)
    x = zeros(size(b));
    r = b;
    p = r;
    k = 0;
    while(norm(r, 'fro') > tau * min(lambdanFz * norm(x, 'fro'), 1) && k < maxiter)
        Ap = Axhandle(p);
        alpha = r(:)' * r(:) / (p(:)' * Ap(:));
        x = x + alpha * p;
        rr0 = r(:)' * r(:);
        r = r - alpha * Ap;
        beta = r(:)' * r(:) / rr0;
        p = r + beta * p;
        k = k + 1;
    end
    output = x;
end

function output = embedded_intr_normal_c(x, eta)
    [n, p] = size(x.main);
    tmp = x.main' * eta;
    tmp = (tmp + tmp') / 2;
    output = zeros(p * (p + 1) / 2, 1);
    r2 = sqrt(2);
    idx = 0;
    for i = 1 : p
        idx = idx + 1;
        output(idx) = tmp(i, i);
    end
    for i = 1 : p
        for j = i + 1 : p
            idx = idx + 1;
            output(idx) = tmp(i, j) * r2;
        end
    end
end

function output = embedded_full_normal_c(x, v)
    [n, p] = size(x.main);
    idx = 0;
    r2 = sqrt(2);
    tmp = zeros(p, p);
    for i = 1 : p
        idx = idx + 1;
        tmp(i, i) = v(idx);
    end
    for i = 1 : p
        for j = i + 1 : p
            idx = idx + 1;
            tmp(i, j) = v(idx) / r2;
            tmp(j, i) = tmp(i, j);
        end
    end
    output = x.main * tmp;
end

function L = vecT(m,n)
    L = zeros(m*n);
    for i = 1:m
        for j = 1:n
            L(n*(i-1)+j,i+m*(j-1)) = 1;
        end
    end
end

function output = Jx_u(x, u, M, Bt_v, Bv, B_lambda, H, t, A)

    [n,r] = size(x.main);
    u = reshape(u,[n,r]);
    Lambdau = Lambda_u(x, u, M, Bv, Bt_v, H);

    %B_lambda = Bv(x,lambda);
    L_u = W_map(x,u,B_lambda);
    tmp3 = -2 * A'*(A*u) - L_u;

    Lambda_tmp = Lambda_u(x,tmp3,M, Bv, Bt_v, H);

    output = u - Lambdau + t*Lambda_tmp;
    output = output(:);

end

function output = Lambda_u(x, u, M, Bv, Bt_v, H)
    Mu = M.*u;
    B_Mu = Bt_v(x,Mu);
    tmp1 =  H\B_Mu;
    tmp2 = Bv(x,tmp1);
    output = Mu - M.*tmp2;
end


function output = W_map(x, z, v) % z\in T_x M and v\in N_x M
    output = -z * x.main'*v - .5 * x.main * (z'*v + v'*z);
end



