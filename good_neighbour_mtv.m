%% Unified good neighbors learning for multi-view data
function [Rn] = good_neighbour_mtv(Z,N,num_closer,num_judge)

K = size(Z,2);

for k = 1:K
    [S_weight{k},S_number{k},S_weight_max{k},S_max{k}] = choose_value(Z{k},num_closer,num_judge);
end

[S_number_new,S_weight_new] = gn(S_number,S_weight,K,N,num_closer);
[S_max_new,S_weight_max_new] = gn(S_max,S_weight_max,K,N,num_closer);

[Rn,Rm] = omp(S_number_new,S_max_new,S_weight_new,S_weight_max_new);

if size(Rn,2) == N-1
    Rn(:,N) = zeros(N,1);
    Rn(N,N-1) = 1;
end
Rn(1:N+1:end) = 0;

if size(Rm,2) == N-1
    Rm(:,N) = zeros(N,1);
    Rm(N,N-1) = 1;
end
Rm(1:N+1:end) = 0;

end

function [S_weight,S_number,S_weight_max,S_max] = choose_value(Z,num_closer,num_judge)

C = abs(Z) + abs(Z');

for i = 1:size(Z,1)
    D(i,:) = C(i,:)/C(i,i);
    
    for j = 1:num_judge
        [p,q] = max(D(i,:));
        S_number_temp(i,j) = q;
        S_weight_temp(i,j) = p;
        D(i,q) = 0;
    end
end

S_max = S_number_temp(:,1:num_closer);
S_weight_max = S_weight_temp(:,1:num_closer);

C = [];
D = [];
S_weight = zeros(size(Z,1),num_closer);
S_number = zeros(size(Z,1),num_closer);

for i = 1:size(Z,1)
    
    C(i,:) = abs(Z(i,:)) + abs(Z(:,i)');
    D(i,:) = C(i,:)/C(i,i);
    
    num = 0;
    
    for j = 1:num_judge
        if num < num_closer
            temp_1 = S_number_temp(i,j);
            temp_2 = S_weight_temp(i,j);
            
            for k = 1:num_judge
                if find(S_number_temp(S_number_temp(temp_1,k),:) == i)
                    num = num + 1;
                    S_number(i,num) = temp_1;
                    S_weight(i,num) = temp_2;
                    break;
                end
            end
        else
            break;
        end  
    end
    
end

for ii = 1:size(Z,1)
    tempa = S_number(ii,:);
    tempindex = tempa~=0;
    tempb = tempa(tempindex);
    tempn = length(tempb);
    
    if tempn ~= num_closer
        temp11 = S_max(ii,:);
        temp22 = S_weight_max(ii,:);
        
        for jj = 1:tempn
            if find(temp11 == tempb(jj))~=0
                tempt = temp11 == tempb(jj);
                temp11(tempt) = [];
                temp22(tempt) = [];
            end
        end
        
        S_number(ii,(tempn+1):end) = temp11(1:(num_closer - tempn));
        S_weight(ii,(tempn+1):end) = temp22(1:(num_closer - tempn));
    end
end

end

function [number_new,weight_new] = gn(number,weight,K,N,num_closer)

number_all = [];
weight_all = [];
for k = 1:K
    number_all = [number_all,number{k}];
    weight_all = [weight_all,weight{k}];
end

[weight_all,index]= sort(weight_all,2,'descend');

for i = 1:N
    temp = number_all(i,:);
    tempindex = index(i,:);
    number_all(i,:) = temp(tempindex);
end

for i = 1:N
    [a, b] = unique(number_all(i,:), 'first');
    result = sortrows([b, a']);
    temp = result(:, 2)';
    number_new(i,:) = temp(1:num_closer);
    temp2 = result(:,1)';
    temp2 = temp2(1:num_closer);
    temp3 = weight_all(i,:);
    weight_new(i,:) = temp3(temp2);
end

end

function [C,D] = omp(S,S1,S_weight,S_weight_max)

C = zeros(size(S,1),size(S,1));
for i=1:size(S,1)
    for j=1:size(S,2)
        C(i,S(i,j))=S_weight(i,j);
    end
end

D = zeros(size(S,1),size(S,1));
for m=1:size(S1,1)
    for n=1:size(S1,2)
        D(m,S1(m,n))=S_weight_max(m,n);
    end
end

end
