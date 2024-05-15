function [x_prox,Act_set,Inact_set] = proximal_l1_diag(b,lambda,r,invdiaglist)
    % diag_list should be a column vector
    if ~iscolumn(invdiaglist)
        invdiaglist=invdiaglist';
    end
    a = abs(b) - lambda*invdiaglist;
    if r < 15
      Act_set = double(a > 0);
    else
      Act_set = (a > 0);
    end
    x_prox = (Act_set.*sign(b)).*a;
    if nargout==3
         Inact_set= (a <= 0);
    end

end

