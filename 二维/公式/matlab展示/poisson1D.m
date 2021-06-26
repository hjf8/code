function poisson1D(a,b,n)
a=0;b=2*pi;n=100;
h=(b-a)/n
x=[a:h:b];
A(1:n+1,1:n+1)=0;
A(1,1) =  1.0;
A(n+1,n+1) =  1.0;
R(1,1) = 0.0;
R(n+1,1)=0.0;
for i=2:n
   R(i,1)=sin(x(i))*h^2;
   A(i,i)=2;
   A(i,i-1)=-1;
   A(i,i+1)=-1;
end
size(A);
size(R);
u = A\R;
%u=gauss(A,R);
%x0=zeros(size(A,1),1);
%[u,n]=myJacobi(A,R,x0,1.0e-6,1000000);
%[u,n]=guaseidel(A,R,x0,1.0e-6,1000000);
%n
%[u, n] = CG(A,R,x0,1.0e-12,100000);
%n
%plot(x,u);

end