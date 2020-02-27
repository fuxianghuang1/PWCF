function [ L] = cl(H,paras)

k     =500;
sigma =paras.sigma;

    % Construct neighborhood graph
    disp('Constructing neighborhood graph...');

        G = L2_distance(H, H);

        [~, ind] = sort(G); 
        for i=1:size(G, 1)
            G(i, ind((2 + k):end, i)) = 0; 
        end
        G = sparse(double(G));
        G = max(G, G');             % Make sure distance matrix is symmetric

     G = G .^ 2;
	 G = G ./ max(max(G));

% disp('Constructing neighborhood graph...');
%         G = L2_distance(X', X');
%         GS=G(1:ns,1:ns);
%         [~, ind] = sort(GS,2);%
%         for i=1:size(GS, 1)
%             GS(i, ind(i,(2 + k):end)) = 0; 
%         end
%         GT=G(ns+1:ns+nt,ns+1:ns+nt);
%         [~, ind] = sort(GT,2); 
%         for i=1:size(GT, 1)
%             GT(i, ind(i,(2 + k):end)) = 0; 
%         end
%         
%         GH= L2_distance(HS,HT);
%         [~, ind] = sort(GH,2); 
%         for i=1:size(GH, 1)
%             GH(i, ind(i,(2 + k):end)) = 0; 
%         end
%         G(1:ns,ns+1:ns+nt)=GH;
%         G(ns+1:ns+nt,1:ns)=GH';
%         G(1:ns,1:ns)=GS;
%         G(ns+1:ns+nt,ns+1:ns+nt)=GT;

        G = sparse(double(G));
        G = max(G, G'); 
        G = G .^ 2;
	    G = G ./ max(max(G));
    
    % Compute weights (W = G)
    disp('Computing weight matrices...');
    
    % Compute Gaussian kernel (heat kernel-based weights)
    G(G ~= 0) = exp(-G(G ~= 0) / (2 * sigma ^ 2));
     D = diag(sum(G, 2));
    
    % Compute Laplacian
    L = D - G;
    L(isnan(L)) = 0; D(isnan(D)) = 0;
	L(isinf(L)) = 0; D(isinf(D)) = 0;

end
