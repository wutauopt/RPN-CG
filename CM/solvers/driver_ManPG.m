
function [xopt, fs, Ds, iter, sparsity, comtime, iter_time] = driver_ManPG(H, option, d_l, V)
% min -Tr(X'*H*X)+ mu*norm(X,1) s.t. X'*X=Ir. X \in R^{n*r}
% mu can be a vector with weighted parameter
% parameters
    % parameters
    n = option.n;
    r = option.r;
    mu = option.mu;
    tol = option.tol;
    maxiter = option.maxiter;
    stop_label = option.stop;
    x0 = option.x0;


    % parameters for line search
    delta = 0.001;
    
    % functions for the optimization problem
    fhandle = @(x)f(x, H, mu);
    gfhandle = @(x)Eucgf(x, H);
    fprox = @prox;
    fcalJ = @calJ;
    
    % functions for the manifold
    fcalA = @calA;
    fcalAstar = @calAstar;
    
    xinitial.main = x0;
    L = 8/d_l^2.*(sin(pi/4))^2 + V;
    t = 1/L; 
    tic
    [xopt, fs, Ds, iter, iter_time] = solver(fhandle, gfhandle, fcalA, fcalAstar, fprox, fcalJ, xinitial, t, tol, delta, maxiter, mu, stop_label, option);
    comtime = toc;
    xopt.main(abs(xopt.main) < 1e-5) = 0;
    sparsity = sum(sum(abs(xopt.main) < 1e-5)) / (n * r);
    xopt = xopt.main;
    fprintf('ManPG:*** Iter ***  Fval *** CPU  **** sparsity *** opt_norm  \n');
    print_format = '     %i     %1.5e    %1.2f        %1.2f        %1.3e       \n';
    fprintf(1,print_format, iter,fs(end), comtime,sparsity,Ds(end));
end

function [xopt, fs, Ds, iter, iter_time] = solver(fhandle, gfhandle, fcalA, fcalAstar, fprox, fcalJ, x0, t, tol, delta, maxiter, mu, stop_label, option)
    err = inf;
    x1 = x0;
    x2 = x1; fs = []; Ds = [];
    [f1, x1] = fhandle(x1);
    gf1 = gfhandle(x1);
    iter = 0;
    fs(iter + 1) = f1;
    [n, p] = size(x0.main);
    Dinitial = zeros(p, p);
    totalbt = 0;
    innertol = max(1e-13, min(1e-11,1e-3*sqrt(tol)*t^2));
    nv = 1;
    while(err > stop_label  && iter < maxiter)
        
        innertol = min(max(1e-30, nv * nv * 1e-8), innertol);

        [v, Dinitial, inneriter] = finddir(x1, gf1, t, fcalA, fcalAstar, fprox, fcalJ, mu, Dinitial, innertol);
        nv = norm(v,'fro');
        alpha = 1;
        x2 = R(x1, alpha * v);
        [f2, x2] = fhandle(x2);
        btiter = 0;
        while(f2 > f1 - delta * alpha * nv^2 && btiter < 3)
            alpha = .5* alpha ;
            x2 = R(x1, alpha * v);
            [f2, x2] = fhandle(x2);
            btiter = btiter + 1;
            totalbt = totalbt + 1;
        end
        gf2 = gfhandle(x2);
        
        iter = iter + 1;
        err = nv;
        Ds(iter) = nv; fs(iter + 1) = f2;
        iter_time(iter) = toc;
        if(mod(iter, option.outputgap) == 0)
            fprintf('iter:%d, f:%e, nv:%e, btiter:%d \n', iter, f1, nv, btiter);
        end
        x1 = x2; f1 = f2; gf1 = gf2;
    end
    fprintf('iter:%d, f:%e, nv:%e, ngf:%e, totalbt:%d\n', iter, f1, nv, norm(gf1, 'fro'), totalbt);
    xopt = x2;
end

function output = R(x, eta)
    [Q,R] = qr(x.main + eta,0);
    [U,S,V] = svd(R);
    output.main = Q*(U*V');
end

function [output, x] = f(x, H, mu)
   x.Hx = H * x.main;
   output = - trace(x.main' * x.Hx)  +  mu * sum(abs(x.main(:)));
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
    output = tmp + tmp';
end

function output = calAstar(Lambda, U) % U \in St(p, n)
    output = U * (Lambda + Lambda');
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
