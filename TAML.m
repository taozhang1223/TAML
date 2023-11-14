clear;
addpath(genpath('datasets'));
addpath(genpath('functions'));

%% Datasets
load('BBC-Sport.mat');      % lambda = 0.01;   alpha = 0.1;   kn = 50;   num_closer = 10£»
% load('ORL.mat');            % lambda = 0.01;   alpha = 0.1;   kn = 50;   num_closer = 10£»
% load('UCI-Digits.mat');     % lambda = 0.1;    alpha = 0.1;   kn = 50;   num_closer = 250£»

%% Parameters
lambda = 0.01; 
alpha = 0.1; 
kn = 50;
num_closer = 10;

num_judge = 2 * num_closer;

%% TAML: Calculate Z
K = size(X,2);
N = size(X{1},2);
cls_num = size(unique(gt),1);

for k = 1:K
    X{k} = NormalizeData(X{k});
end

for k = 1:K
    D{k} = L2_distance_1(X{k}, X{k});
end

mu1 = 1e-4;        max_mu1 = 10e10;        pho_mu1 = 2;
mu2 = 1e-4;        max_mu2 = 10e10;        pho_mu2 = 2;
epson = 1e-7;      max_iter = 200;

% Initialize

for k = 1:K
    Z{k} = zeros(N,N);
    E{k} = zeros(size(X{k},1),N);
    S{k} = zeros(N,N);
    M{k} = zeros(N,N);
    A{k} = zeros(size(X{k},1),N);
    B{k} = zeros(N,N);
end

b = zeros(N*N*K,1);
m = zeros(N*N*K,1);
sX = [N, N, K];

iter = 0;
Isconverg = 0;

% Update

while(Isconverg == 0)
    
    disp(['Iter£º    ' num2str(iter +1 )]);
    
    % 1 update Z^k
    for k = 1:K
        tmp = (X{k}'*A{k} + mu1*X{k}'*X{k} - mu1*X{k}'*E{k} - B{k})./mu2 +  M{k};
        Z{k} = inv(eye(N,N)+ (mu1/mu2)*X{k}'*X{k})*tmp;
    end
    
    % 2 update E^k
    for k = 1:K
        F = X{k}-X{k}*Z{k}+A{k}/mu1;
        E{k} = solve_l1l2(F,lambda/mu1);
    end
    
    % 3 update A^k
    for k = 1:K
        A{k} = A{k} + mu1*(X{k}-X{k}*Z{k}-E{k});
    end
    
    % 4 update S^k
    SD = [];
    for k = 1:K
        SD{k} = D{k}-alpha*M{k};
        [dumb, idx] = sort(SD{k}, 2);
        for i = 1:N
            id = idx(i,2:kn+2);
            di = SD{k}(i, id);
            S{k}(i,id) = (di(kn+1)-di)/(kn*di(kn+1)-sum(di(1:kn))+eps);
        end
        S{k} = (S{k}+S{k}')/2;
    end
    
    % 5 update M
    Z_tensor = cat(3, Z{:,:});
    B_tensor = cat(3, B{:,:});
    S_tensor = cat(3, S{:,:});
    z = Z_tensor(:);
    b = B_tensor(:);
    s = S_tensor(:);
    
    % twist-version
    temp1 =(alpha*s+b+mu2*z)/(alpha+mu2);
    temp2 = 1/(alpha+mu2);
    [m,objV] = wshrinkObj(temp1,temp2,sX,0,3);
    M_tensor = reshape(m, sX);
    
    % 6 update B
    b = b + mu2*(z - m);
    
    % Coverge condition
    
    Isconverg = 1;
    for k = 1:K
        if (norm(X{k}-X{k}*Z{k}-E{k},inf)>epson) 
            Isconverg = 0;
        end
        
        M{k} = M_tensor(:,:,k);
        B_tensor = reshape(b, sX);
        B{k} = B_tensor(:,:,k);
        if (norm(Z{k}-M{k},inf)>epson)
            Isconverg = 0;
        end
    end
    
    if (iter > max_iter)
        Isconverg  = 1;
    end
    
    iter = iter + 1;
    mu1 = min(mu1*pho_mu1, max_mu1);
    mu2 = min(mu2*pho_mu2, max_mu2);
end

%% TAML: Calculate W
[Rn] = good_neighbour_mtv(Z,N,num_closer,num_judge);
W = abs(Rn)+abs(Rn');

%% Result
C = SpectralClustering(W,cls_num);
[~,NMI,~] = compute_nmi(gt,C);
ACC = Accuracy(C,double(gt));

disp(['NMI = ' num2str(NMI)]);
disp(['ACC = ' num2str(ACC)]);