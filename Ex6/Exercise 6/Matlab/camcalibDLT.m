function P=camcalibDLT(x_world, x_im)
    % Inputs: 
    %   x_world: World coordinates with shape (point_id, coordinates)
    %   x_im: Image coordinates with shape (point_id, coordinates)
    %
    % Outputs:
    %   P: Camera projection matrix with shape (3,4)
    
    % Create the matrix A 
    %%-your-code-starts-here-%%

    %%-your-code-ends-here-%%
    
    % Perform homogeneous least squares fitting.
    % The best solution is given by the eigenvector of 
    % A.T*A with the smallest eigenvalue.
    %%-your-code-starts-here-%%

    %%-your-code-ends-here-%%

    % Reshape the eigenvector into a projection matrix P
    %P = (reshape(ev,4,3))'; % here ev is the eigenvector above
    P = [0 0 0 0;0 0 0 0;0 0 0 1];  % remove this and uncomment the above
end