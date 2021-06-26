function F = bertsekas(x)
%x = ones(15,1)';
X = xiabiao();
F = zeros(15,1);
gama=0.5;
d = [0.1,0.2,0.3,0.4,0.5]';
for i = 1:15
    if i>=1 & i<=5
         F(i) = 2*(1+gama)*g(x(i))+2*gama*g(x(X(i,3)))+2*gama*g(x(X(i,4))) ...,
         +gama*g(x(X(i,3))+x(X(i,4)))+10*g(x(i)+x(X(i,3))+x(X(i,4)))+g(x(i)+x(X(i,4))) ...,
         +10*g(x(i)+x(X(i,1))+x(X(i,4)))+g(x(i)+x(X(i,1)))+10*g(x(i)+x(X(i,1))+x(X(i,2)))-x(i+10);
    else if i>=6&i<=10
            F(i) = (3+2*gama)*g(x(i))+3*gama*g(x(X(i,1)))+10*g(x(i)+x(X(i,1)))+10*g(x(i)+x(X(i,4)))-x(i+5);
        else
            F(i) = x(i-10)+x(i-5)-d(i-10);
        end
   end
end