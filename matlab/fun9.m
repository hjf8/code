n=3;p=4;
X=rand(n,p);
A=rand(n);
A=A'+A;
B=X'*A*X;
for j=1:length(A)
    for i=j:n
        C((n-j/2)*(j-1)+i)=A(i,j);
    end
end
for q=1:length(B)
    for m=q:p
        D((p-q/2)*(q-1)+m)=B(m,q);
    end
end
A
C
B
D

