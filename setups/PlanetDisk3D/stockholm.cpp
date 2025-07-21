// === Fargo-style damping zone (updated to match y_inf/y_sup logic) ===
/*
                real dampingzone = 1.07;
                real taudamp = 0.1;

                real R_inf = rmin * pow(dampingzone, 2.0/3.0);
                real R_sup = rmax * pow(dampingzone, -2.0/3.0);

                real ramp = 0.0;
                if(R < R_inf) {
                    ramp = (R_inf - R) / (R_inf - rmin);
                    ramp *= ramp;  // parabolic ramp
                } else if(R > R_sup) {
                    ramp = (R - R_sup) / (rmax - R_sup);
                    ramp *= ramp;
                }

                if(ramp > 0.0) {
                    real tau_fargo = taudamp * sqrt(R * R * R);
                    real taud = tau_fargo / ramp;

                    real rho = Vc(RHO,k,j,i);
                    real rhoTarget = sigma0 * pow(R, -sigmaSlope);
                    Uc(RHO,k,j,i) = (rho * taud + rhoTarget * dt) / (dt + taud);

                    //#ifndef ISOTHERMAL
                    //real cs2 = h0 * h0 * pow(R, 2 * flaringIndex - 1.0);
                    //real eTarget = cs2 * rhoTarget / (gamma - 1.0);
                    //real eint = Uc(ENG,k,j,i);
                    //#endif

                    // Momentum components
                    real vx1 = Vc(VX1,k,j,i);
                    real vx2 = Vc(VX2,k,j,i);
                    real vx3 = Vc(VX3,k,j,i);
                    real vx2Target = 0.0;

                    if(!isFargo) {
                        vx2Target = Vk * sqrt(1.0 - (1.0 + sigmaSlope - 2.0 * flaringIndex)
                                              * h0 * h0 * pow(R, 2.0 * flaringIndex)) - omega * R;
                    }

                    real momx1 = rho * vx1;
                    real momx2 = rho * vx2;
                    real momx3 = rho * vx3;

                    real momx1Target = 0.0;
                    real momx2Target = rhoTarget * vx2Target;
                    real momx3Target = 0.0;

                    Uc(MX1,k,j,i) = (momx1 * taud + momx1Target * dt) / (dt + taud);
                    Uc(MX2,k,j,i) = (momx2 * taud + momx2Target * dt) / (dt + taud);
                    Uc(MX3,k,j,i) = (momx3 * taud + momx3Target * dt) / (dt + taud);
                    //#ifndef ISOTHERMAL
                    //Uc(ENG,k,j,i) = (eint * taud + eTarget * dt) / (dt + taud);
                    //#endif

                }
/*
