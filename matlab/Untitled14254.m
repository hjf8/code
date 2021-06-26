a=0;b=1;n=1;lamuda=5;
syms s
d=ones(n,1);
for i=1:n
    %p=5-(exp(b^2)-1)/b
    y=5*exp(-s)*cos(s) + (s - 1)/(s^2 - 2*s + 2) - (exp(s - 1)*(sin(1) - cos(1) + s*cos(1)))/(s^2 - 2*s + 2);
    tt=a:0.1:b;
    y=vpa(norm(subs(y,s,tt),Inf),20)
    
    %y=vpa(int(abs(y),s,0,1),10)
    p=b^(2*i+1)*exp(b^2)/factorial(i+1)   %k-kn
    q=1/(5-(exp(b^2)-1)/b)%inv(lamuda-k)
    d=(1/(1/q-p))*p*q*y
    %p=lamuda-(exp(b^2)-1)/b;
    %d(i)=x*p*q;
end
d