
function [xopt, fs, Ds, iter, sparsity, comtime, iter_time] = driver_rpncg(H, option,d_l, V)
    % parameters
    n = option.n;
    r = option.r;
    mu = option.mu;
    tol = option.tol;
    maxiter = option.maxiter;
    stop_label = option.stop;
    x0 = option.x0;

    % parameters
    params.vartheta = .01; params.gamma = .01;
    params.rho1 = 0.001; params.rho2 = 0.5;  % line search parameter
    params.w1 = 1.1; params.w2 = 0.9;
    params.theta = 0.5;
    params.kappa = 0.1;
    params.max_innit = 200;
    tau = 100;

    
    % functions for the optimization problem
    fhandle = @(x)f(x, H, mu);
    gfhandle = @(x)Eucgf(x, H);
    Bv = @(x,v)x.main*v;
    fprox = @prox;
    fcalJ = @calJ;
    
    % functions for the manifold
    fcalA = @calA;
    fcalAstar = @calAstar;
    
    x.main = x0;
    L = 8/d_l^2.*(sin(pi/4))^2 + V;
    t = 1/L;
    tic
    [xopt, fs, Ds, iter, iter_time] = solver(H, fhandle, gfhandle, fcalA, fcalAstar, fprox, fcalJ, Bv, x, t, tau, tol, maxiter, mu, stop_label, params, option);
    comtime = toc;
    xopt.main(abs(xopt.main) < 1e-5) = 0;
    sparsity = sum(sum(abs(xopt.main) < 1e-5)) / (n * r);
    xopt = xopt.main;
    fprintf('RPN-CG:*** Iter ***  Fval *** CPU  **** sparsity *** opt_norm  \n');
    print_format = '     %i     %1.5e    %1.2f        %1.2f        %1.3e       \n';
    fprintf(1,print_format, iter,fs(end), comtime, sparsity, Ds(end));
end

function [xopt, fs, Ds, iter, iter_time] = solver(H, fhandle, gfhandle, fcalA, fcalAstar, fprox, fcalJ, Bv, x0, t, tau, tol, maxiter, mu, stop_label, params, option)
    err = inf;
    x1 = x0;
    x2 = x1; fs = []; Ds = [];
    [f1, x1] = fhandle(x1);
    gf1 = gfhandle(x1);
    iter = 0;
    fs(iter + 1) = f1;
    [n, r] = size(x0.main);
    params.n = n;
    params.r = r;
    Dinitial = zeros(r, r);
    status = 0; flag = 0;
    innertol = max(1e-13, min(1e-11,1e-3*sqrt(tol)*t^2));
    nv = 1;
    t0 = t;
  
    while(err > stop_label &&  iter < maxiter)

        innertol = min(max(1e-30, nv * nv * 1e-8), innertol);
        if status >= 4
            innertol = min(max(1e-30, nv * nv * nv), innertol);
        end

        [v, Dinitial, inneriter] = finddir(x1, gf1, t, fcalA, fcalAstar, fprox, fcalJ, mu, Dinitial, innertol);
        nv = norm(v,'fro');
        lambda = Dinitial;
        Blambda = Bv(x1,lambda);

        x1.x_v = x1.main + v;
        id1 = find(x1.x_v~=0 & abs(x1.main) >= nv);
        id2 = find(x1.x_v==0 | abs(x1.main) < nv);
        num_1 = length(id1); % number of nonzero
        num_2 = length(id2); % number of zero
        params.max_innit = round(num_1*1.2);

        v_bar = v(id1);
        v_hat = v(id2);

        tmp11 = x1.main'*Blambda;
        BB_handle = @(y)BBB(y,x1,H,Blambda,tmp11);
        % compute w by tCG
        [w, inner_it, status] = tCG(BB_handle,v,nv,v_bar,v_hat,x1,gf1,id1,id2,t,tau,mu,params);
        u = zeros(n,r);
        u(id2) = v_hat;
        u(id1) = v_bar + w;
        nu = norm(u,'fro');

        % update t
        if (4 + 1 / t) * nu < nv || status == 0
            t = max(t0, params.w2 * t);
        elseif status <= 4
            t = params.w1 * t;
        end

        if(mod(iter+1, option.outputgap) == 0)
            fprintf('iter:%d, inner_it:%d,status:%d,t:%e, tau:%e\n', iter+1,inner_it,status,t,tau);
            fprintf('iter:%d, f:%e, nv:%e, nu:%e, nonzero:%d, zero:%d\n', iter+1, f1, nv, nu,num_1,num_2);
        end

        if (status >= 5 || flag == 1)
            flag = flag + 1;
            if flag == 1
                x2 = R(x1, u);
                [f2,x2] = fhandle(x2);
                xold = x1; fold = f1; uold = u; nuold = nu; nvold = nv;
            else
                flag = 0;
                x2 = R(x1, u);
                [f2, x2] = fhandle(x2);
                if f2 > fold - params.rho1 * nvold^2
                    % linesearch
                    num_linesearch2 = 0;
                    alpha = 1;
                    x2 = R(xold, alpha * uold);
                    [f2, x2] = fhandle(x2);
                    btiter = 0;
                    while f2 > fold - params.rho1 * alpha * nuold^2 && btiter < 3
                        alpha = params.rho2 * alpha;
                        num_linesearch2 = num_linesearch2 + 1;
                        btiter = btiter + 1;
                        x2 = R(xold, alpha * uold);
                        [f2,x2] = fhandle(x2);
                    end
                end
            end
        else
            flag = 0;
            % linesearch
            num_linesearch1 = 0;
            alpha = 1;
            x2 = R(x1, alpha * u);
            [f2, x2] = fhandle(x2);
            btiter = 0;
            normDsquared = nu^2;
            while f2 > f1 - params.rho1 * alpha * normDsquared && btiter < 3
                alpha = params.rho2 * alpha;
                num_linesearch1 = num_linesearch1 + 1;
                btiter = btiter + 1;
                x2 = R(x1, alpha * u);
                [f2,x2] = fhandle(x2);
            end
        end
        gf2 = gfhandle(x2);
        err = min(nv,nu);
        iter = iter + 1;
        fs(iter+1) = f2;
        Ds(iter) = nv;
        iter_time(iter) = toc;
        if(mod(iter, option.outputgap) == 0)
            fprintf('iter:%d, f:%e, nv:%e, nu:%e,  itertime:%f,innertol:%e \n', iter, f1, nv, nu,  iter_time(iter), innertol);
        end
        x1 = x2; f1 = f2; gf1 = gf2;
    end
    
    fprintf('iter:%d, f:%e, nv:%e,nu:%e, ngf:%e, t:%e\n', iter, f1, nv,nu, norm(gf1, 'fro'), t);
    xopt = x2;
end

function output = R(x, eta)
    [Q,R] = qr(x.main + eta,0);[U,S,V] = svd(R);
    output.main = Q*(U*V');
end

function [output, x] = f(x, H, mu)
   x.Hx = H * x.main;
   x.hx =  mu * sum(abs(x.main(:)));
   output = - trace(x.main' * x.Hx)  + x.hx;
end

function [output, x] = Eucgf(x, H)
   output = -2 * x.Hx;
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
   tmp = Z' * U;
   output = -0.5 * (tmp + tmp');
end

function output = calAstar(Lambda, U) % U \in St(p, n)
   output = - U * ((Lambda + Lambda')/2);
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

function [output] = BBB(y,x,H,Blambda,tmp11)
    tmp = y'*Blambda;
    tt = H*y;
    tmp1 = -2 * tt;
    tmp2 = y*tmp11;
    tmp3 = x.main*((tmp + tmp')/2);
    output = tmp1 + tmp2 + tmp3; 
end

