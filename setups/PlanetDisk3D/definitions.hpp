#define COMPONENTS   3
#define DIMENSIONS   3       // Switch from 2D to 3D
#define GEOMETRY     POLAR   // POLAR in 3D corresponds to Cylindrical (R, Z, phi)

#define THERMALDIFFUSION // This is handled by TDiffusion in the .ini file
//#define ISOTHERMAL
//#define STOCKHOLM        // Enable the Stockholm damping zone
//#define BETACOOLING
#define UNSTRAT          // Add this flag for the unstratified disk model

//#define SYMMETRIC_BC
#define NOPERT_BC
