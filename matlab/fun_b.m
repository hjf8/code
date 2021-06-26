function B_vec=fun_b(n)
A=rand(n,n);
A=(A+A')/2;
X=rand(n,n);
A_vec=symvec(A);
B_vec=zeros(size(A_vec));
for i=1:n
    for j=i:n
        XX=X(:,i)*X(:,j)';
        XX=tril(XX)+triu(XX,1)';
        XX=symvec(XX);
        B_vec((n-0.5*i)*(i-1)+j)=XX'*A_vec;
    end
end
disp(B_vec)