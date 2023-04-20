function X=trianglin(P1,P2,x1,x2)
    
    % Inputs:
    %   P1: Projection matrices for image 1 with shape (3,4)
    %   P2: Projection matrices for image 2 with shape (3,4)
    %   x1: Image coordinates for a point in image 1
    %   x2: Image coordinates for a point in image 2
    %
    % Outputs:
    %   X: Triangulated world coordinates
    
    % Form A and get the least squares solution from the eigenvector 
    % corresponding to the smallest eigenvalue
    %%-your-code-starts-here-%%

    %%-your-code-ends-here-%%
    
    X = [0;0;0;1];  % remove me

end
