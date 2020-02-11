
#include <Maestro.H>
#include <Maestro_F.H>

using namespace amrex;

void
Maestro::PPM (const Box& bx, 
              Array4<const Real> const s,
              Array4<const Real> const u,
              Array4<const Real> const v,
#if (AMREX_SPACEDIM == 3)
              Array4<const Real> const w,
#endif
              Array4<Real> const Ip,
              Array4<Real> const Im,
              const Box& domainBox,
              const Vector<BCRec>& bcs,
              const Real* dx,
              const bool is_umac, const int comp, const int bccomp)
{
    // timer for profiling
    BL_PROFILE_VAR("Maestro::PPM()",PPM);

    // constant used in Colella 2008
    const Real C = 1.25;
    Real rel_eps = 0.0;
    get_rel_eps(&rel_eps);

    const int n = comp;

    const Real hx = dx[0];
    const Real hy = dx[1];
#if (AMREX_SPACEDIM == 3)
    const Real hz = dx[2];
#endif

    const Real dt_local = dt;

    /////////////
    // x-dir
    /////////////

    const int ilo = domainBox.loVect()[0];
    const int ihi = domainBox.hiVect()[0];
    int bclo = bcs[bccomp].lo()[0];
    int bchi = bcs[bccomp].hi()[0];

    if (ppm_type == 1) {

        AMREX_PARALLEL_FOR_3D(bx, i, j, k, 
        {
            // Compute van Leer slopes in x-direction 

            // sm
            Real dsvl_l = 0.0;
            Real dsvl_r = 0.0;

            // left side 
            Real dsc = 0.5 * (s(i,j,k,n) - s(i-2,j,k,n));
            Real dsl = 2.0 * (s(i-1,j,k,n) - s(i-2,j,k,n));
            Real dsr = 2.0 * (s(i,j,k,n) - s(i-1,j,k,n));
            if (dsl*dsr > 0.0) 
                dsvl_l = copysign(1.0,dsc)*min(fabs(dsc),min(fabs(dsl),fabs(dsr)));

            // right side
            dsc = 0.5 * (s(i+1,j,k,n) - s(i-1,j,k,n));
            dsl = 2.0 * (s(i,j,k,n) - s(i-1,j,k,n));
            dsr = 2.0 * (s(i+1,j,k,n) - s(i,j,k,n));
            if (dsl*dsr > 0.0) 
                dsvl_r = copysign(1.0,dsc)*min(fabs(dsc),min(fabs(dsl),fabs(dsr)));

            // Interpolate s to x-edges.
            Real sm = 0.5*(s(i,j,k,n)+s(i-1,j,k,n)) - (dsvl_r-dsvl_l)/6.0;
            
            // Make sure sedge lies in between adjacent cell-centered values.
            sm = max(sm,min(s(i,j,k,n),s(i-1,j,k,n)));
            sm = min(sm,max(s(i,j,k,n),s(i-1,j,k,n)));

            // sp
            dsvl_l = 0.0;
            dsvl_r = 0.0;

            // left side
            dsc = 0.5 * (s(i+1,j,k,n) - s(i-1,j,k,n));
            dsl = 2.0 * (s(i,j,k,n) - s(i-1,j,k,n));
            dsr = 2.0 * (s(i+1,j,k,n) - s(i,j,k,n));
            if (dsl*dsr > 0.0) 
                dsvl_l = copysign(1.0,dsc)*min(fabs(dsc),min(fabs(dsl),fabs(dsr)));

            // right side
            dsc = 0.5 * (s(i+2,j,k,n) - s(i,j,k,n));
            dsl = 2.0 * (s(i+1,j,k,n) - s(i,j,k,n));
            dsr = 2.0 * (s(i+2,j,k,n) - s(i+1,j,k,n));
            if (dsl*dsr > 0.0) 
                dsvl_r = copysign(1.0,dsc)*min(fabs(dsc),min(fabs(dsl),fabs(dsr)));
            
            // Interpolate s to x-edges.
            Real sp = 0.5*(s(i+1,j,k,n)+s(i,j,k,n)) - (dsvl_r-dsvl_l) / 6.0;
            
            // Make sure sedge lies in between adjacent cell-centered values.
            sp = max(sp,min(s(i+1,j,k,n),s(i,j,k,n)));
            sp = min(sp,max(s(i+1,j,k,n),s(i,j,k,n)));

            // save for later 
            Real sedgel = sp;
            Real sedger = sm;
            
            // Modify using quadratic limiters.
            if ((sp-s(i,j,k,n))*(s(i,j,k,n)-sm) <= 0.0) {
                sp = s(i,j,k,n);
                sm = s(i,j,k,n);
            } else if (fabs(sp-s(i,j,k,n)) >= 2.0*fabs(sm-s(i,j,k,n))) {
                sp = 3.0*s(i,j,k,n) - 2.0*sm;
            } else if (fabs(sm-s(i,j,k,n)) >= 2.0*fabs(sp-s(i,j,k,n))) {
                sm = 3.0*s(i,j,k,n) - 2.0*sp;
            }

            // Different stencil needed for x-component of EXT_DIR and HOEXTRAP adv_bc's.
            if (i == ilo) {
                if (bclo == EXT_DIR || bclo == HOEXTRAP) {

                    // The value in the first cc ghost cell represents the edge value.
                    sm = s(i-1,j,k,n);
                    
                    // Use a modified stencil to get sp on the first interior edge.
                    sp = -0.2 *s(i-1,j,k,n) 
                        + 0.75*s(i,j,k,n) 
                        + 0.5 *s(i+1,j,k,n) 
                        - 0.05*s(i+2,j,k,n);
                    
                    // Make sure sedge lies in between adjacent cell-centered values.
                    sp = max(sp,min(s(i+1,j,k,n),s(i,j,k,n)));
                    sp = min(sp,max(s(i+1,j,k,n),s(i,j,k,n)));
                }

            } else if (i == ilo+1) {
                if (bclo == EXT_DIR || bclo == HOEXTRAP) {
                    
                    // Use a modified stencil to get sedge on the first interior edge.
                    sm = -0.2 *s(i-2,j,k,n) 
                        + 0.75*s(i-1,j,k,n) 
                        + 0.5 *s(i,j,k,n) 
                        - 0.05*s(i+1,j,k,n);
                    
                    // Make sure sedge lies in between adjacent cell-centered values.
                    sm = max(sm,min(s(i,j,k,n),s(i-1,j,k,n)));
                    sm = min(sm,max(s(i,j,k,n),s(i-1,j,k,n)));

                    // reset sp on second interior edge
                    sp = sedgel;
                    
                    // Modify using quadratic limiters.
                    if ((sp-s(i,j,k,n))*(s(i,j,k,n)-sm) <= 0.0) {
                        sp = s(i,j,k,n);
                        sm = s(i,j,k,n);
                    } else if (fabs(sp-s(i,j,k,n)) >= 2.0*fabs(sm-s(i,j,k,n))) {
                        sp = 3.0*s(i,j,k,n) - 2.0*sm;
                    } else if (fabs(sm-s(i,j,k,n)) >= 2.0*fabs(sp-s(i,j,k,n))) {
                        sm = 3.0*s(i,j,k,n) - 2.0*sp;
                    }
                }

            } else if (i == ihi) {
                if (bchi == EXT_DIR || bchi == HOEXTRAP) {

                    // The value in the first cc ghost cell represents the edge value.
                    sp = s(i+1,j,k,n);

                    // Use a modified stencil to get sm on the first interior edge.
                    sm = -0.2 *s(i+1,j,k,n) 
                        + 0.75*s(i,j,k,n) 
                        + 0.5 *s(i-1,j,k,n) 
                        - 0.05*s(i-2,j,k,n);
                    
                    // Make sure sm lies in between adjacent cell-centered values.
                    sm = max(sm,min(s(i-1,j,k,n),s(i,j,k,n)));
                    sm = min(sm,max(s(i-1,j,k,n),s(i,j,k,n)));
                }

            } else if (i == ihi-1) {
                if (bchi == EXT_DIR  || bchi == HOEXTRAP) {
                    
                    // Use a modified stencil to get sp on the first interior edge.
                    sp = -0.2 *s(i+2,j,k,n) 
                        + 0.75*s(i+1,j,k,n) 
                        + 0.5 *s(i,j,k,n) 
                        - 0.05*s(i-1,j,k,n);
                    
                    // Make sure sp lies in between adjacent cell-centered values.
                    sp = max(sp,min(s(i,j,k,n),s(i+1,j,k,n)));
                    sp = min(sp,max(s(i,j,k,n),s(i+1,j,k,n)));

                    // reset sm on second interior edge
                    sm = sedger;

                    // Modify using quadratic limiters.
                    if ((sp-s(i,j,k,n))*(s(i,j,k,n)-sm) <= 0.0) {
                        sp = s(i,j,k,n);
                        sm = s(i,j,k,n);
                    } else if (fabs(sp-s(i,j,k,n)) >= 2.0*fabs(sm-s(i,j,k,n))) {
                        sp = 3.0*s(i,j,k,n) - 2.0*sm;
                    } else if (fabs(sm-s(i,j,k,n)) >= 2.0*fabs(sp-s(i,j,k,n))) {
                        sm = 3.0*s(i,j,k,n) - 2.0*sp;
                    }
                }
            }

            ////////////////////////////////////
            // Compute x-component of Ip and Im.
            ////////////////////////////////////
            Real s6 = 6.0*s(i,j,k,n) - 3.0*(sm+sp);

            if (is_umac) {

                // u is MAC velocity -- use edge-based indexing
                Real sigma = fabs(u(i+1,j,k)) * dt_local / hx;
                if (u(i+1,j,k) > rel_eps) {
                    Ip(i,j,k,0) = sp - 0.5*sigma*(sp-sm-(1.0-2.0/3.0*sigma)*s6);
                } else {
                    Ip(i,j,k,0) = s(i,j,k,n);
                }

                sigma = fabs(u(i,j,k)) * dt_local / hx;
                if (u(i,j,k) < -rel_eps) {
                    Im(i,j,k,0) = sm + 0.5*sigma*(sp-sm+(1.0-2.0/3.0*sigma)*s6);
                } else {
                    Im(i,j,k,0) = s(i,j,k,n);
               }

            } else {

                Real sigma = fabs(u(i,j,k))*dt_local/hx;
                if (u(i,j,k) > rel_eps) {
                    Ip(i,j,k,0) = sp - 0.5*sigma*(sp-sm-(1.0-2.0/3.0*sigma)*s6);
                } else {
                    Ip(i,j,k,0) = s(i,j,k,n);
                }

                sigma = fabs(u(i,j,k))*dt_local/hx;
                if (u(i,j,k) < -rel_eps) {
                    Im(i,j,k,0) = sm + 0.5*sigma*(sp-sm+(1.0-2.0/3.0*sigma)*s6);
                } else {
                    Im(i,j,k,0) = s(i,j,k,n);
                }
            }
        });

    } else if (ppm_type == 2) {

        AMREX_PARALLEL_FOR_3D(bx, i, j, k, 
        {
            // -1
            // Interpolate s to x-edges.
            Real sedgel = (7.0/12.0)*(s(i-2,j,k,n)+s(i-1,j,k,n)) 
                    - (1.0/12.0)*(s(i-3,j,k,n)+s(i,j,k,n));

            // Limit sedge.
            if ((sedgel-s(i-2,j,k,n))*(s(i-1,j,k,n)-sedgel) < 0.0) {
                Real D2  = 3.0*(s(i-2,j,k,n)-2.0*sedgel+s(i-1,j,k,n));
                Real D2L = s(i-3,j,k,n)-2.0*s(i-2,j,k,n)+s(i-1,j,k,n);
                Real D2R = s(i-2,j,k,n)-2.0*s(i-1,j,k,n)+s(i,j,k,n);
                Real sgn = copysign(1.0,D2);
                Real D2LIM = sgn*max(min(C*sgn*D2L,min(C*sgn*D2R,sgn*D2)),0.0);
                sedgel = 0.5*(s(i-2,j,k,n)+s(i-1,j,k,n)) - D2LIM/6.0;
            }

            // 0
            // Interpolate s to x-edges.
            Real sedge = (7.0/12.0)*(s(i-1,j,k,n)+s(i,j,k,n)) 
                    - (1.0/12.0)*(s(i-2,j,k,n)+s(i+1,j,k,n));

            // Limit sedge.
            if ((sedge-s(i-1,j,k,n))*(s(i,j,k,n)-sedge) < 0.0) { 
                Real D2  = 3.0*(s(i-1,j,k,n)-2.0*sedge+s(i,j,k,n));
                Real D2L = s(i-2,j,k,n)-2.0*s(i-1,j,k,n)+s(i,j,k,n);
                Real D2R = s(i-1,j,k,n)-2.0*s(i,j,k,n)+s(i+1,j,k,n);
                Real sgn = copysign(1.0,D2);
                Real D2LIM = sgn*max(min(C*sgn*D2L,min(C*sgn*D2R,sgn*D2)),0.0);
                sedge = 0.5*(s(i-1,j,k,n)+s(i,j,k,n)) - D2LIM/6.0;
            }

            // +1
            // Interpolate s to x-edges.
            Real sedger = (7.0/12.0)*(s(i,j,k,n)+s(i+1,j,k,n)) 
                    - (1.0/12.0)*(s(i-1,j,k,n)+s(i+2,j,k,n));

            // Limit sedge.
            if ((sedger-s(i,j,k,n))*(s(i+1,j,k,n)-sedger) < 0.0) {
                Real D2  = 3.0*(s(i,j,k,n)-2.0*sedger+s(i+1,j,k,n));
                Real D2L = s(i-1,j,k,n)-2.0*s(i,j,k,n)+s(i+1,j,k,n);
                Real D2R = s(i,j,k,n)-2.0*s(i+1,j,k,n)+s(i+2,j,k,n);
                Real sgn = copysign(1.0,D2);
                Real D2LIM = sgn*max(min(C*sgn*D2L,min(C*sgn*D2R,sgn*D2)),0.0);
                sedger = 0.5*(s(i,j,k,n)+s(i+1,j,k,n)) - D2LIM/6.0;
            }

            // +2
            // Interpolate s to x-edges.
            Real sedgerr = (7.0/12.0)*(s(i+1,j,k,n)+s(i+2,j,k,n)) 
                    - (1.0/12.0)*(s(i,j,k,n)+s(i+3,j,k,n));

            // Limit sedge.
            if ((sedgerr-s(i+1,j,k,n))*(s(i+2,j,k,n)-sedgerr) < 0.0) {
                Real D2  = 3.0*(s(i+1,j,k,n)-2.0*sedgerr+s(i+2,j,k,n));
                Real D2L = s(i,j,k,n)-2.0*s(i+1,j,k,n)+s(i+2,j,k,n);
                Real D2R = s(i+1,j,k,n)-2.0*s(i+2,j,k,n)+s(i+3,j,k,n);
                Real sgn = copysign(1.0,D2);
                Real D2LIM = sgn*max(min(C*sgn*D2L,min(C*sgn*D2R,sgn*D2)),0.0);
                sedgerr = 0.5*(s(i+1,j,k,n)+s(i+2,j,k,n)) - D2LIM/6.0;
            }

            // use Colella 2008 limiters
            // This is a new version of the algorithm
            // to eliminate sensitivity to roundoff.

            Real alphap = sedger-s(i,j,k,n);
            Real alpham = sedge-s(i,j,k,n);
            bool bigp = fabs(alphap) > 2.0*fabs(alpham);
            bool bigm = fabs(alpham) > 2.0*fabs(alphap);
            bool extremum = false;

            if (alpham*alphap >= 0.0) {
                extremum = true;
            } else if (bigp || bigm) {
            
                // Possible extremum. We look at cell centered values and face
                // centered values for a change in sign in the differences adjacent to
                // the cell. We use the pair of differences whose minimum magnitude is the
                // largest, and thus least susceptible to sensitivity to roundoff.
                Real dafacem = sedge - sedgel;
                Real dafacep = sedgerr - sedger;
                Real dabarm = s(i,j,k,n) - s(i-1,j,k,n);
                Real dabarp = s(i+1,j,k,n) - s(i,j,k,n);
                Real dafacemin = min(fabs(dafacem),fabs(dafacep));
                Real dabarmin = min(fabs(dabarm),fabs(dabarp));
                Real dachkm = 0.0;
                Real dachkp = 0.0;

                if (dafacemin >= dabarmin) {
                    dachkm = dafacem;
                    dachkp = dafacep;
                } else {
                    dachkm = dabarm;
                    dachkp = dabarp;
                }
                extremum = (dachkm*dachkp <= 0.0);
            }

            if (extremum) {
                Real D2  = 6.0*(alpham + alphap);
                Real D2L = s(i-2,j,k,n)-2.0*s(i-1,j,k,n)+s(i,j,k,n);
                Real D2R = s(i,j,k,n)-2.0*s(i+1,j,k,n)+s(i+2,j,k,n);
                Real D2C = s(i-1,j,k,n)-2.0*s(i,j,k,n)+s(i+1,j,k,n);
                Real sgn = copysign(1.0,D2);
                Real D2LIM = max(min(sgn*D2,min(C*sgn*D2L,min(C*sgn*D2R,C*sgn*D2C))),0.0);
                Real D2ABS = max(fabs(D2),1.e-10);
                alpham = alpham*D2LIM/D2ABS;
                alphap = alphap*D2LIM/D2ABS;
            } else {
                if (bigp) {
                    Real sgn = copysign(1.0,alpham);
                    Real amax = -alphap*alphap / (4.0*(alpham + alphap));
                    Real delam = s(i-1,j,k,n) - s(i,j,k,n);
                    if (sgn*amax >= sgn*delam) {
                        if (sgn*(delam - alpham) >= 1.e-10) {
                            alphap = -2.0*delam - 2.0*sgn*sqrt(delam*delam - delam*alpham);
                        } else {
                            alphap = -2.0*alpham;
                        }
                    }
                }
                if (bigm) {
                    Real sgn = copysign(1.0,alphap);
                    Real amax = -alpham*alpham / (4.0*(alpham + alphap));
                    Real delap = s(i+1,j,k,n) - s(i,j,k,n);
                    if (sgn*amax >= sgn*delap) {
                        if (sgn*(delap - alphap) >= 1.e-10) {
                            alpham = (-2.0*delap -2.0*sgn*sqrt(delap*delap - delap*alphap));
                        } else {
                            alpham = -2.0*alphap;
                        }
                    }
                }
            }

            Real sm = s(i,j,k,n) + alpham;
            Real sp = s(i,j,k,n) + alphap;

            // different stencil needed for x-component of EXT_DIR and HOEXTRAP adv_bc's
            if (bclo == EXT_DIR || bclo == HOEXTRAP) {
                if (i == ilo) {
                    // The value in the first cc ghost cell represents the edge value.
                    sm = s(i-1,j,k,n);

                    // use a modified stencil to get sedge on the first interior edge
                    sp = -0.2 *s(i-1,j,k,n) 
                        + 0.75*s(i,j,k,n) 
                        + 0.5 *s(i+1,j,k,n) 
                        - 0.05*s(i+2,j,k,n);

                    // make sure sedge lies in between adjacent cell-centered values
                    sp = max(sp,min(s(i+1,j,k,n),s(i,j,k,n)));
                    sp = min(sp,max(s(i+1,j,k,n),s(i,j,k,n)));

                } else if (i == ilo+1) {

                    sedgel = s(i-2,j,k,n);

                    // use a modified stencil to get sedge on the first interior edge
                    sedge = -0.2 *s(i-2,j,k,n) 
                           + 0.75*s(i-1,j,k,n) 
                           + 0.5 *s(i,j,k,n) 
                           - 0.05*s(i+1,j,k,n);

                    // make sure sedge lies in between adjacent cell-centered values
                    sedge = max(sedge,min(s(i,j,k,n),s(i-1,j,k,n)));
                    sedge = min(sedge,max(s(i,j,k,n),s(i-1,j,k,n)));

                } else if (i == ilo+2) {

                    // use a modified stencil to get sedge on the first interior edge
                    sedgel = -0.2 *s(i-3,j,k,n) 
                            + 0.75*s(i-2,j,k,n) 
                            + 0.5 *s(i-1,j,k,n) 
                            - 0.05*s(i,j,k,n);

                    // make sure sedge lies in between adjacent cell-centered values
                    sedgel = max(sedgel,min(s(i-1,j,k,n),s(i-2,j,k,n)));
                    sedgel = min(sedgel,max(s(i-1,j,k,n),s(i-2,j,k,n)));
                }

                // Apply Colella 2008 limiters to compute sm and sp in the second
                // and third inner cells.
                if (i == ilo+1 || i == ilo+2) {

                    alphap = sedger-s(i,j,k,n);
                    alpham = sedge-s(i,j,k,n);
                    bigp = fabs(alphap) > 2.0*fabs(alpham);
                    bigm = fabs(alpham) > 2.0*fabs(alphap);
                    extremum = false;

                    if (alpham*alphap >= 0.0) {
                        extremum = true;
                    } else if (bigp || bigm) {
                        // Possible extremum. We look at cell centered values and face
                        // centered values for a change in sign in the differences adjacent to
                        // the cell. We use the pair of differences whose minimum magnitude is
                        // the largest, and thus least susceptible to sensitivity to roundoff.
                        Real dafacem = sedge - sedgel;
                        Real dafacep = sedgerr - sedger;
                        Real dabarm = s(i,j,k,n) - s(i-1,j,k,n);
                        Real dabarp = s(i+1,j,k,n) - s(i,j,k,n);
                        Real dafacemin = min(fabs(dafacem),fabs(dafacep));
                        Real dabarmin = min(fabs(dabarm),fabs(dabarp));
                        Real dachkm = 0.0;
                        Real dachkp = 0.0;

                        if (dafacemin >= dabarmin) {
                            dachkm = dafacem;
                            dachkp = dafacep;
                        } else {
                            dachkm = dabarm;
                            dachkp = dabarp;
                        }
                        extremum = (dachkm*dachkp <= 0.0);
                    }

                    if (extremum) {
                        Real D2  = 6.0*(alpham + alphap);
                        Real D2L = s(i-2,j,k,n)-2.0*s(i-1,j,k,n)+s(i,j,k,n);
                        Real D2R = s(i,j,k,n)-2.0*s(i+1,j,k,n)+s(i+2,j,k,n);
                        Real D2C = s(i-1,j,k,n)-2.0*s(i,j,k,n)+s(i+1,j,k,n);
                        Real sgn = copysign(1.0,D2);
                        Real D2LIM = max(min(sgn*D2,min(C*sgn*D2L,min(C*sgn*D2R,C*sgn*D2C))),0.0);
                        Real D2ABS = max(fabs(D2),1.e-10);
                        alpham = alpham*D2LIM/D2ABS;
                        alphap = alphap*D2LIM/D2ABS;
                    } else {
                        if (bigp) {
                            Real sgn = copysign(1.0,alpham);
                            Real amax = -alphap*alphap / (4.0*(alpham + alphap));
                            Real delam = s(i-1,j,k,n) - s(i,j,k,n);
                            if (sgn*amax >= sgn*delam) {
                                if (sgn*(delam - alpham) >= 1.e-10) {
                                    alphap = (-2.0*delam - 2.0*sgn*sqrt(delam*delam - delam*alpham));
                                } else {
                                    alphap = -2.0*alpham;
                                }
                            }
                        }
                        if (bigm) {
                            Real sgn = copysign(1.0,alphap);
                            Real amax = -alpham*alpham / (4.0*(alpham + alphap));
                            Real delap = s(i+1,j,k,n) - s(i,j,k,n);
                            if (sgn*amax >= sgn*delap) {
                                if (sgn*(delap - alphap) >= 1.e10) {
                                    alpham = (-2.0*delap - 2.0*sgn*sqrt(delap*delap - delap*alphap));
                                } else {
                                    alpham = -2.0*alphap;
                                }
                            }
                        }
                    }

                    sm = s(i,j,k,n) + alpham;
                    sp = s(i,j,k,n) + alphap;
                }
            }

            if (bchi == EXT_DIR  || bchi == HOEXTRAP) {
                if (i == ihi) {
                    // The value in the first cc ghost cell represents the edge value.
                    sp = s(i+1,j,k,n);

                    // Use a modified stencil to get sedge on the first interior edge.
                    sm = -0.2 *s(i+1,j,k,n) 
                        + 0.75*s(i,j,k,n)
                        + 0.5 *s(i-1,j,k,n) 
                        - 0.05*s(i-2,j,k,n);
      
                    // Make sure sedge lies in between adjacent cell-centered values.
                    sm = max(sm,min(s(i-1,j,k,n),s(i,j,k,n)));
                    sm = min(sm,max(s(i-1,j,k,n),s(i,j,k,n)));

                } else if (i == ihi-1) {
                    // Use a modified stencil to get sedge on the first interior edge.
                    sedger = -0.2 *s(i+2,j,k,n) 
                            + 0.75*s(i+1,j,k,n) 
                            + 0.5 *s(i,j,k,n) 
                            - 0.05*s(i-1,j,k,n);
                    
                    // Make sure sedge lies in between adjacent cell-centered values.
                    sedger = max(sedger,min(s(i,j,k,n),s(i+1,j,k,n)));
                    sedger = min(sedger,max(s(i,j,k,n),s(i+1,j,k,n)));

                } else if (i == ihi-2) {
                    // Use a modified stencil to get sedge on the first interior edge.
                    sedgerr = -0.2 *s(i+3,j,k,n) 
                             + 0.75*s(i+2,j,k,n) 
                             + 0.5 *s(i+1,j,k,n) 
                             - 0.05*s(i,j,k,n);
                    
                    // Make sure sedge lies in between adjacent cell-centered values.
                    sedgerr = max(sedgerr,min(s(i+1,j,k,n),s(i+2,j,k,n)));
                    sedgerr = min(sedgerr,max(s(i+1,j,k,n),s(i+2,j,k,n)));
                }

                // Apply Colella 2008 limiters to compute sm and sp in the second
                // and third inner cells.
                if (i == ihi-1 || i == ihi-2) {

                    alphap = sedger-s(i,j,k,n);
                    alpham = sedge-s(i,j,k,n);
                    bigp = fabs(alphap) > 2.0*fabs(alpham);
                    bigm = fabs(alpham) > 2.0*fabs(alphap);
                    extremum = false;

                    if (alpham*alphap >= 0.0) {
                        extremum = true;
                    } else if (bigp || bigm) {
                        // Possible extremum. We look at cell centered values and face
                        // centered values for a change in sign in the differences adjacent to
                        // the cell. We use the pair of differences whose minimum magnitude is
                        // the largest, and thus least susceptible to sensitivity to roundoff.
                        Real dafacem = sedge - sedgel;
                        Real dafacep = sedgerr - sedger;
                        Real dabarm = s(i,j,k,n) - s(i-1,j,k,n);
                        Real dabarp = s(i+1,j,k,n) - s(i,j,k,n);
                        Real dafacemin = min(fabs(dafacem),fabs(dafacep));
                        Real dabarmin = min(fabs(dabarm),fabs(dabarp));
                        Real dachkm = 0.0;
                        Real dachkp = 0.0;
                        if (dafacemin >= dabarmin) {
                            dachkm = dafacem;
                            dachkp = dafacep;
                        } else {
                            dachkm = dabarm;
                            dachkp = dabarp;
                        }
                        extremum = (dachkm*dachkp <= 0.0);
                    }

                    if (extremum) {
                        Real D2  = 6.0*(alpham + alphap);
                        Real D2L = s(i-2,j,k,n)-2.0*s(i-1,j,k,n)+s(i,j,k,n);
                        Real D2R = s(i,j,k,n)-2.0*s(i+1,j,k,n)+s(i+2,j,k,n);
                        Real D2C = s(i-1,j,k,n)-2.0*s(i,j,k,n)+s(i+1,j,k,n);
                        Real sgn = copysign(1.0,D2);
                        Real D2LIM = max(min(sgn*D2,min(C*sgn*D2L,min(C*sgn*D2R,C*sgn*D2C))),0.0);
                        Real D2ABS = max(fabs(D2),1.e-10);
                        alpham = alpham*D2LIM/D2ABS;
                        alphap = alphap*D2LIM/D2ABS;
                    } else {
                        if (bigp) {
                            Real sgn = copysign(1.0,alpham);
                            Real amax = -alphap*alphap / (4.0*(alpham + alphap));
                            Real delam = s(i-1,j,k,n) - s(i,j,k,n);
                            if (sgn*amax >= sgn*delam) {
                                if (sgn*(delam - alpham) >= 1.e-10) {
                                    alphap = (-2.0*delam - 2.0*sgn*sqrt(delam*delam - delam*alpham));
                                } else {
                                    alphap = -2.0*alpham;
                                }
                            }
                        }
                        if (bigm) {
                            Real sgn = copysign(1.0,alphap);
                            Real amax = -alpham*alpham / (4.0*(alpham + alphap));
                            Real delap = s(i+1,j,k,n) - s(i,j,k,n);
                            if (sgn*amax >= sgn*delap) {
                                if (sgn*(delap - alphap) >= 1.e-10) {
                                    alpham = (-2.0*delap - 2.0*sgn*sqrt(delap*delap - delap*alphap));
                                } else {
                                    alpham = -2.0*alphap;
                               }
                            }
                        }
                    }

                    sm = s(i,j,k,n) + alpham;
                    sp = s(i,j,k,n) + alphap;
                }
            }

            ////////////////////////////////////
            // Compute x-component of Ip and Im.
            ////////////////////////////////////

            Real s6 = 6.0*s(i,j,k,n) - 3.0*(sm+sp);

            if (is_umac) {

                // u is MAC velocity -- use edge-based indexing
                Real sigma = fabs(u(i+1,j,k))*dt_local/hx;
                
                if (u(i+1,j,k) > rel_eps) {
                    Ip(i,j,k,0) = sp - 0.5*sigma*(sp-sm-(1.0-2.0/3.0*sigma)*s6);
                } else {
                    Ip(i,j,k,0) = s(i,j,k,n);
                }

                sigma = fabs(u(i,j,k))*dt_local/hx;

                if (u(i,j,k) < -rel_eps) {
                    Im(i,j,k,0) = sm + 0.5*sigma*(sp-sm+(1.0-2.0/3.0*sigma)*s6);
                } else {
                    Im(i,j,k,0) = s(i,j,k,n);
                }
            } else {

                Real sigma = fabs(u(i,j,k))*dt_local/hx;

                if (u(i,j,k) > rel_eps) {
                    Ip(i,j,k,0) = sp - 0.5*sigma*(sp-sm-(1.0-2.0/3.0*sigma)*s6);
                } else {
                    Ip(i,j,k,0) = s(i,j,k,n);
                }

                sigma = fabs(u(i,j,k))*dt_local/hx;

                if (u(i,j,k) < -rel_eps) {
                    Im(i,j,k,0) = sm + 0.5*sigma*(sp-sm+(1.0-2.0/3.0*sigma)*s6);
                } else {
                    Im(i,j,k,0) = s(i,j,k,n);
                }
            }
        });
    }

    /////////////
    // y-dir
    /////////////

    const int jlo = domainBox.loVect()[1];
    const int jhi = domainBox.hiVect()[1];
    bclo = bcs[bccomp].lo()[1];
    bchi = bcs[bccomp].hi()[1];

    if (ppm_type == 1) {

        AMREX_PARALLEL_FOR_3D(bx, i, j, k, 
        {
            // Compute van Leer slopes in y-direction.

            // sm
            Real dsvl_l = 0.0;
            Real dsvl_r = 0.0;

            // left side
            Real dsc = 0.5 * (s(i,j,k,n) - s(i,j-2,k,n));
            Real dsl = 2.0 * (s(i,j-1,k,n) - s(i,j-2,k,n));
            Real dsr = 2.0 * (s(i,j,k,n) - s(i,j-1,k,n));
            if (dsl*dsr > 0.0) 
                dsvl_l = copysign(1.0,dsc)*min(fabs(dsc),min(fabs(dsl),fabs(dsr)));

            // right side
            dsc = 0.5 * (s(i,j+1,k,n) - s(i,j-1,k,n));
            dsl = 2.0 * (s(i,j,k,n) - s(i,j-1,k,n));
            dsr = 2.0 * (s(i,j+1,k,n) - s(i,j,k,n));
            if (dsl*dsr > 0.0) 
                dsvl_r = copysign(1.0,dsc)*min(fabs(dsc),min(fabs(dsl),fabs(dsr)));

            // Interpolate s to y-edges.
            Real sm = 0.5*(s(i,j,k,n)+s(i,j-1,k,n)) - (dsvl_r-dsvl_l)/6.0;

            // Make sure sedge lies in between adjacent cell-centered values.
            sm = max(sm,min(s(i,j,k,n),s(i,j-1,k,n)));
            sm = min(sm,max(s(i,j,k,n),s(i,j-1,k,n)));

            // sp
            dsvl_l = 0.0;
            dsvl_r = 0.0;

            // left side
            dsc = 0.5 * (s(i,j+1,k,n) - s(i,j-1,k,n));
            dsl = 2.0 * (s(i,j,k,n) - s(i,j-1,k,n));
            dsr = 2.0 * (s(i,j+1,k,n) - s(i,j,k,n));
            if (dsl*dsr > 0.0) 
                dsvl_l = copysign(1.0,dsc)*min(fabs(dsc),min(fabs(dsl),fabs(dsr)));

            // right side
            dsc = 0.5 * (s(i,j+2,k,n) - s(i,j,k,n));
            dsl = 2.0 * (s(i,j+1,k,n) - s(i,j,k,n));
            dsr = 2.0 * (s(i,j+2,k,n) - s(i,j+1,k,n));
            if (dsl*dsr > 0.0) 
                dsvl_r = copysign(1.0,dsc)*min(fabs(dsc),min(fabs(dsl),fabs(dsr)));

            // Interpolate s to y-edges.
            Real sp = 0.5*(s(i,j+1,k,n)+s(i,j,k,n)) - (dsvl_r-dsvl_l) / 6.0;
           
            // Make sure sedge lies in between adjacent cell-centered values.
            sp = max(sp,min(s(i,j+1,k,n),s(i,j,k,n)));
            sp = min(sp,max(s(i,j+1,k,n),s(i,j,k,n)));

            // save for later 
            Real sedgel = sp;
            Real sedger = sm;

            // Modify using quadratic limiters.
            if ((sp-s(i,j,k,n))*(s(i,j,k,n)-sm) <= 0.0) {
                sp = s(i,j,k,n);
                sm = s(i,j,k,n);
            } else if (fabs(sp-s(i,j,k,n)) >= 2.0*fabs(sm-s(i,j,k,n))) {
                sp = 3.0*s(i,j,k,n) - 2.0*sm;
            } else if (fabs(sm-s(i,j,k,n)) >= 2.0*fabs(sp-s(i,j,k,n))) {
                sm = 3.0*s(i,j,k,n) - 2.0*sp;
            }
            
            // Different stencil needed for y-component of EXT_DIR and HOEXTRAP adv_bc's.
            if (j == jlo) {
                if (bclo == EXT_DIR || bclo == HOEXTRAP) {
                  
                    // The value in the first cc ghost cell represents the edge value.
                    sm = s(i,j-1,k,n);
                   
                    // Use a modified stencil to get sedge on the first interior edge.
                    sp = -0.2 *s(i,j-1,k,n) 
                        + 0.75*s(i,j,k,n) 
                        + 0.5 *s(i,j+1,k,n) 
                        - 0.05*s(i,j+2,k,n);
                
                    // Make sure sp lies in between adjacent cell-centered values.
                    sp = max(sp,min(s(i,j+1,k,n),s(i,j,k,n)));
                    sp = min(sp,max(s(i,j+1,k,n),s(i,j,k,n)));
                }

            } else if (j == jlo+1) {
                if (bclo == EXT_DIR || bclo == HOEXTRAP) {

                    // Use a modified stencil to get sm on the first interior edge.
                    sm = -0.2 *s(i,j-2,k,n) 
                        + 0.75*s(i,j-1,k,n) 
                        + 0.5 *s(i,j,k,n) 
                        - 0.05*s(i,j+1,k,n);
                   
                    // Make sure sedge lies in between adjacent cell-centered values.
                    sm = max(sm,min(s(i,j,k,n),s(i,j-1,k,n)));
                    sm = min(sm,max(s(i,j,k,n),s(i,j-1,k,n)));

                    // reset sp on second interior edge
                    sp = sedgel;
                    
                    // Modify using quadratic limiters.
                    if ((sp-s(i,j,k,n))*(s(i,j,k,n)-sm) <= 0.0) {
                        sp = s(i,j,k,n);
                        sm = s(i,j,k,n);
                    } else if (fabs(sp-s(i,j,k,n)) >= 2.0*fabs(sm-s(i,j,k,n))) {
                        sp = 3.0*s(i,j,k,n) - 2.0*sm;
                    } else if (fabs(sm-s(i,j,k,n)) >= 2.0*fabs(sp-s(i,j,k,n))) {
                        sm = 3.0*s(i,j,k,n) - 2.0*sp;
                    }
                }

            } else if (j == jhi) {
                if (bchi == EXT_DIR  || bchi == HOEXTRAP) {
            
                    // The value in the first cc ghost cell represents the edge value.
                    sp = s(i,j+1,k,n);
             
                    // Use a modified stencil to get sm on the first interior edge.
                    sm = -0.2 *s(i,j+1,k,n) 
                        + 0.75*s(i,j,k,n) 
                        + 0.5 *s(i,j-1,k,n) 
                        - 0.05*s(i,j-2,k,n);
                  
                    // Make sure sm lies in between adjacent cell-centered values.
                    sm = max(sm,min(s(i,j-1,k,n),s(i,j,k,n)));
                    sm = min(sm,max(s(i,j-1,k,n),s(i,j,k,n)));
                }

            } else if (j == jhi-1) {
                if (bchi == EXT_DIR  || bchi == HOEXTRAP) {

                    // Use a modified stencil to get sp on the first interior edge.
                    sp = -0.2 *s(i,j+2,k,n) 
                        + 0.75*s(i,j+1,k,n) 
                        + 0.5 *s(i,j,k,n) 
                        - 0.05*s(i,j-1,k,n);
                    
                    // Make sure sedge lies in between adjacent cell-centered values.
                    sp = max(sp,min(s(i,j,k,n),s(i,j+1,k,n)));
                    sp = min(sp,max(s(i,j,k,n),s(i,j+1,k,n)));

                    // reset sm on second interior edge
                    sm = sedger;

                    // Modify using quadratic limiters.
                    if ((sp-s(i,j,k,n))*(s(i,j,k,n)-sm) <= 0.0) {
                        sp = s(i,j,k,n);
                        sm = s(i,j,k,n);
                    } else if (fabs(sp-s(i,j,k,n)) >= 2.0*fabs(sm-s(i,j,k,n))) {
                        sp = 3.0*s(i,j,k,n) - 2.0*sm;
                    } else if (fabs(sm-s(i,j,k,n)) >= 2.0*fabs(sp-s(i,j,k,n))) {
                        sm = 3.0*s(i,j,k,n) - 2.0*sp;
                    }
                }
            }

            ////////////////////////////////////
            // Compute y-component of Ip and Im.
            ////////////////////////////////////
            Real s6 = 6.0*s(i,j,k,n) - 3.0*(sm+sp);

            if (is_umac) {

                // v is MAC velocity -- use edge-based indexing
                Real sigma = fabs(v(i,j+1,k))*dt_local/hy;
                if (v(i,j+1,k) > rel_eps) {
                    Ip(i,j,k,1) = sp - 0.5*sigma*(sp-sm-(1.0-2.0/3.0*sigma)*s6);
                } else {
                    Ip(i,j,k,1) = s(i,j,k,n);
                }

                sigma = fabs(v(i,j,k))*dt_local/hy;
                if (v(i,j,k) < -rel_eps) {
                    Im(i,j,k,1) = sm + 0.5*sigma*(sp-sm+(1.0-2.0/3.0*sigma)*s6);
                } else {
                    Im(i,j,k,1) = s(i,j,k,n);
                }

            } else {

                Real sigma = fabs(v(i,j,k))*dt_local/hy;
                if (v(i,j,k) > rel_eps) {
                    Ip(i,j,k,1) = sp - 0.5*sigma*(sp-sm-(1.0-2.0/3.0*sigma)*s6);
                } else {
                    Ip(i,j,k,1) = s(i,j,k,n);
                }

                sigma = fabs(v(i,j,k))*dt_local/hy;
                if (v(i,j,k) < -rel_eps) {
                    Im(i,j,k,1) = sm + 0.5*sigma*(sp-sm+(1.0-2.0/3.0*sigma)*s6);
                } else {
                    Im(i,j,k,1) = s(i,j,k,n);
                }
            }
        });

    } else if (ppm_type == 2) {

        AMREX_PARALLEL_FOR_3D(bx, i, j, k, 
        {
            // -1
            // Interpolate s to y-edges.
            Real sedgel = (7.0/12.0)*(s(i,j-2,k,n)+s(i,j-1,k,n)) 
                    - (1.0/12.0)*(s(i,j-3,k,n)+s(i,j,k,n));
            
            // Limit sedge.
            if ((sedgel-s(i,j-2,k,n))*(s(i,j-1,k,n)-sedgel) < 0.0) {
                Real D2  = 3.0*(s(i,j-2,k,n)-2.0*sedgel+s(i,j-1,k,n));
                Real D2L = s(i,j-3,k,n)-2.0*s(i,j-2,k,n)+s(i,j-1,k,n);
                Real D2R = s(i,j-2,k,n)-2.0*s(i,j-1,k,n)+s(i,j,k,n);
                Real sgn = copysign(1.0,D2);
                Real D2LIM = sgn*max(min(C*sgn*D2L,min(C*sgn*D2R,sgn*D2)),0.0);
                sedgel = 0.5*(s(i,j-2,k,n)+s(i,j-1,k,n)) - D2LIM/6.0;
            }

            // 0
            // Interpolate s to y-edges.
            Real sedge = (7.0/12.0)*(s(i,j-1,k,n)+s(i,j,k,n)) 
                    - (1.0/12.0)*(s(i,j-2,k,n)+s(i,j+1,k,n));
            
            // Limit sedge.
            if ((sedge-s(i,j-1,k,n))*(s(i,j,k,n)-sedge) < 0.0) {
                Real D2  = 3.0*(s(i,j-1,k,n)-2.0*sedge+s(i,j,k,n));
                Real D2L = s(i,j-2,k,n)-2.0*s(i,j-1,k,n)+s(i,j,k,n);
                Real D2R = s(i,j-1,k,n)-2.0*s(i,j,k,n)+s(i,j+1,k,n);
                Real sgn = copysign(1.0,D2);
                Real D2LIM = sgn*max(min(C*sgn*D2L,min(C*sgn*D2R,sgn*D2)),0.0);
                sedge = 0.5*(s(i,j-1,k,n)+s(i,j,k,n)) - D2LIM/6.0;
            }

            // +1
            // Interpolate s to y-edges.
            Real sedger = (7.0/12.0)*(s(i,j,k,n)+s(i,j+1,k,n)) 
                    - (1.0/12.0)*(s(i,j-1,k,n)+s(i,j+2,k,n));
            
            // Limit sedge.
            if ((sedger-s(i,j,k,n))*(s(i,j+1,k,n)-sedger) < 0.0) {
                Real D2  = 3.0*(s(i,j,k,n)-2.0*sedger+s(i,j+1,k,n));
                Real D2L = s(i,j-1,k,n)-2.0*s(i,j,k,n)+s(i,j+1,k,n);
                Real D2R = s(i,j,k,n)-2.0*s(i,j+1,k,n)+s(i,j+2,k,n);
                Real sgn = copysign(1.0,D2);
                Real D2LIM = sgn*max(min(C*sgn*D2L,min(C*sgn*D2R,sgn*D2)),0.0);
                sedger = 0.5*(s(i,j,k,n)+s(i,j+1,k,n)) - D2LIM/6.0;
            }

            // +2
            // Interpolate s to y-edges.
            Real sedgerr = (7.0/12.0)*(s(i,j+1,k,n)+s(i,j+2,k,n)) 
                    - (1.0/12.0)*(s(i,j,k,n)+s(i,j+3,k,n));
            
            // Limit sedge.
            if ((sedgerr-s(i,j+1,k,n))*(s(i,j+2,k,n)-sedgerr) < 0.0) {
                Real D2  = 3.0*(s(i,j+1,k,n)-2.0*sedgerr+s(i,j+2,k,n));
                Real D2L = s(i,j,k,n)-2.0*s(i,j+1,k,n)+s(i,j+2,k,n);
                Real D2R = s(i,j+1,k,n)-2.0*s(i,j+2,k,n)+s(i,j+3,k,n);
                Real sgn = copysign(1.0,D2);
                Real D2LIM = sgn*max(min(C*sgn*D2L,min(C*sgn*D2R,sgn*D2)),0.0);
                sedgerr = 0.5*(s(i,j+1,k,n)+s(i,j+2,k,n)) - D2LIM/6.0;
            }

            // Use Colella 2008 limiters.
            // This is a new version of the algorithm
            // to eliminate sensitivity to roundoff.
            Real alphap = sedger-s(i,j,k,n);
            Real alpham = sedge-s(i,j,k,n);
            bool bigp = fabs(alphap) > 2.0*fabs(alpham);
            bool bigm = fabs(alpham) > 2.0*fabs(alphap);
            bool extremum = false;

            if (alpham*alphap >= 0.0) {
                extremum = true;
            } else if (bigp || bigm) {
                // Possible extremum. We look at cell centered values and face
                // centered values for a change in sign in the differences adjacent to
                // the cell. We use the pair of differences whose minimum magnitude is the
                // largest, and thus least susceptible to sensitivity to roundoff.
                Real dafacem = sedge - sedgel;
                Real dafacep = sedgerr - sedger;
                Real dabarm = s(i,j,k,n) - s(i,j-1,k,n);
                Real dabarp = s(i,j+1,k,n) - s(i,j,k,n);
                Real dafacemin = min(fabs(dafacem),fabs(dafacep));
                Real dabarmin= min(fabs(dabarm),fabs(dabarp));
                Real dachkm = 0.0;
                Real dachkp = 0.0;
                if (dafacemin>=dabarmin) {
                    dachkm = dafacem;
                    dachkp = dafacep;
                } else {
                    dachkm = dabarm;
                    dachkp = dabarp;
                }
                extremum = (dachkm*dachkp <= 0.0);
            }

            if (extremum) {
                Real D2  = 6.0*(alpham + alphap);
                Real D2L = s(i,j-2,k,n)-2.0*s(i,j-1,k,n)+s(i,j,k,n);
                Real D2R = s(i,j,k,n)-2.0*s(i,j+1,k,n)+s(i,j+2,k,n);
                Real D2C = s(i,j-1,k,n)-2.0*s(i,j,k,n)+s(i,j+1,k,n);
                Real sgn = copysign(1.0,D2);
                Real D2LIM = max(min(sgn*D2,min(C*sgn*D2L,min(C*sgn*D2R,C*sgn*D2C))),0.0);
                Real D2ABS = max(fabs(D2),1.e-10);
                alpham = alpham*D2LIM/D2ABS;
                alphap = alphap*D2LIM/D2ABS;
            } else {
                if (bigp) {
                    Real sgn = copysign(1.0,alpham);
                    Real amax = -alphap*alphap / (4.0*(alpham + alphap));
                    Real delam = s(i,j-1,k,n) - s(i,j,k,n);
                    if (sgn*amax >= sgn*delam) {
                        if (sgn*(delam - alpham) >= 1.e-10) {
                            alphap = (-2.0*delam - 2.0*sgn*sqrt(delam*delam - delam*alpham));
                        } else {
                            alphap = -2.0*alpham;
                        }
                    }
                }
                if (bigm) {
                    Real sgn = copysign(1.0,alphap);
                    Real amax = -alpham*alpham / (4.0*(alpham + alphap));
                    Real delap = s(i,j+1,k,n) - s(i,j,k,n);
                    if (sgn*amax >= sgn*delap) {
                        if (sgn*(delap - alphap) >= 1.e-10) {
                            alpham = (-2.0*delap - 2.0*sgn*sqrt(delap*delap - delap*alphap));
                        } else {
                            alpham = -2.0*alphap;
                        }
                    }
                }
            }

            Real sm = s(i,j,k,n) + alpham;
            Real sp = s(i,j,k,n) + alphap;

            // Different stencil needed for y-component of EXT_DIR and HOEXTRAP adv_bc's.
            if (bclo == EXT_DIR || bclo == HOEXTRAP) {
                if (j == jlo) {
                    // The value in the first cc ghost cell represents the edge value.
                    sm = s(i,j-1,k,n);
                    sedge = s(i,j-1,k,n);
                   
                    // Use a modified stencil to get sedge on the first interior edge.
                    sp = -0.2 *s(i,j-1,k,n) 
                        + 0.75*s(i,j,k,n) 
                        + 0.5 *s(i,j+1,k,n) 
                        - 0.05*s(i,j+2,k,n);
                    
                    // Make sure sedge lies in between adjacent cell-centered values.
                    sp = max(sp,min(s(i,j+1,k,n),s(i,j,k,n)));
                    sp = min(sp,max(s(i,j+1,k,n),s(i,j,k,n)));

                } else if (j == jlo+1) {

                    sedgel = s(i,j-2,k,n);
                    
                    // Use a modified stencil to get sedge on the first interior edge.
                    sedge = -0.2 *s(i,j-2,k,n) 
                           + 0.75*s(i,j-1,k,n) 
                           + 0.5 *s(i,j,k,n) 
                           - 0.05*s(i,j+1,k,n);
                    
                    // Make sure sedge lies in between adjacent cell-centered values.
                    sedge = max(sedge,min(s(i,j,k,n),s(i,j-1,k,n)));
                    sedge = min(sedge,max(s(i,j,k,n),s(i,j-1,k,n)));

                } else if (j == jlo+2) {
                    // Use a modified stencil to get sedge on the first interior edge.
                    sedgel = -0.2*s(i,j-3,k,n) 
                           + 0.75*s(i,j-2,k,n) 
                           + 0.5 *s(i,j-1,k,n) 
                           - 0.05*s(i,j,k,n);
                    
                    // Make sure sedge lies in between adjacent cell-centered values.
                    sedgel = max(sedgel,min(s(i,j-1,k,n),s(i,j-2,k,n)));
                    sedgel = min(sedgel,max(s(i,j-1,k,n),s(i,j-2,k,n)));
                }

                // Apply Colella 2008 limiters to compute sm and sp in the second
                // and third inner cells.
                if (j == jlo+1 || j == jlo+2) {

                    alphap = sedger-s(i,j,k,n);
                    alpham = sedge-s(i,j,k,n);
                    bigp = fabs(alphap) > 2.0*fabs(alpham);
                    bigm = fabs(alpham) > 2.0*fabs(alphap);
                    extremum = false;

                    if (alpham*alphap >= 0.0) {
                        extremum = true;
                    } else if (bigp || bigm) {
                        
                        // Possible extremum. We look at cell centered values and face
                        // centered values for a change in sign in the differences adjacent to
                        // the cell. We use the pair of differences whose minimum magnitude is
                        // the largest, and thus least susceptible to sensitivity to roundoff.
                        Real dafacem = sedge - sedgel;
                        Real dafacep = sedgerr - sedger;
                        Real dabarm = s(i,j,k,n) - s(i,j-1,k,n);
                        Real dabarp = s(i,j+1,k,n) - s(i,j,k,n);
                        Real dafacemin = min(fabs(dafacem),fabs(dafacep));
                        Real dabarmin= min(fabs(dabarm),fabs(dabarp));
                        Real dachkm = 0.0;
                        Real dachkp = 0.0;
                        if (dafacemin>=dabarmin) {
                            dachkm = dafacem;
                            dachkp = dafacep;
                        } else {
                            dachkm = dabarm;
                            dachkp = dabarp;
                        }
                        extremum = (dachkm*dachkp <= 0.0);
                    }

                    if (extremum) {
                        Real D2  = 6.0*(alpham + alphap);
                        Real D2L = s(i,j-2,k,n)-2.0*s(i,j-1,k,n)+s(i,j,k,n);
                        Real D2R = s(i,j,k,n)-2.0*s(i,j+1,k,n)+s(i,j+2,k,n);
                        Real D2C = s(i,j-1,k,n)-2.0*s(i,j,k,n)+s(i,j+1,k,n);
                        Real sgn = copysign(1.0,D2);
                        Real D2LIM = max(min(sgn*D2,min(C*sgn*D2L,min(C*sgn*D2R,C*sgn*D2C))),0.0);
                        Real D2ABS = max(fabs(D2),1.e-10);
                        alpham = alpham*D2LIM/D2ABS;
                        alphap = alphap*D2LIM/D2ABS;
                    } else {
                        if (bigp) {
                            Real sgn = copysign(1.0,alpham);
                            Real amax = -alphap*alphap / (4.0*(alpham + alphap));
                            Real delam = s(i,j-1,k,n) - s(i,j,k,n);
                            if (sgn*amax >= sgn*delam) {
                                if (sgn*(delam - alpham) >= 1.e-10) {
                                    alphap = (-2.0*delam - 2.0*sgn*sqrt(delam*delam - delam*alpham));
                                } else {
                                    alphap = -2.0*alpham;
                                }
                            }
                        }
                        if (bigm) {
                            Real sgn = copysign(1.0,alphap);
                            Real amax = -alpham*alpham / (4.0*(alpham + alphap));
                            Real delap = s(i,j+1,k,n) - s(i,j,k,n);
                            if (sgn*amax >= sgn*delap) {
                                if (sgn*(delap - alphap) >= 1.e-10) {
                                    alpham = (-2.0*delap - 2.0*sgn*sqrt(delap*delap - delap*alphap));
                                } else {
                                    alpham = -2.0*alphap;
                                }
                            }
                        }
                    }

                    sm = s(i,j,k,n) + alpham;
                    sp = s(i,j,k,n) + alphap;
                }
            }

            if (bchi == EXT_DIR  || bchi == HOEXTRAP) {
                if (j == jhi) {
                    // The value in the first cc ghost cell represents the edge value.
                    sp = s(i,j+1,k,n);
                    
                    // Use a modified stencil to get sedge on the first interior edge.
                    sm  = -0.2*s(i,j+1,k,n) 
                        + 0.75*s(i,j,k,n) 
                        + 0.5 *s(i,j-1,k,n) 
                        - 0.05*s(i,j-2,k,n);
                    
                    // Make sure sedge lies in between adjacent cell-centered values.
                    sm = max(sm,min(s(i,j-1,k,n),s(i,j,k,n)));
                    sm = min(sm,max(s(i,j-1,k,n),s(i,j,k,n)));

                } else if (j == jhi-1) {

                    sedgerr = s(i,j+2,k,n);
                    
                    // Use a modified stencil to get sedge on the first interior edge.
                    sedger = -0.2*s(i,j+2,k,n) 
                           + 0.75*s(i,j+1,k,n) 
                           + 0.5 *s(i,j,k,n) 
                           - 0.05*s(i,j-1,k,n);
                    
                    // Make sure sedge lies in between adjacent cell-centered values.
                    sedger = max(sedger,min(s(i,j,k,n),s(i,j+1,k,n)));
                    sedger = min(sedger,max(s(i,j,k,n),s(i,j+1,k,n)));

                } else if (j == jhi-2) {
                    // Use a modified stencil to get sedge on the first interior edge.
                    sedgerr = -0.2*s(i,j+3,k,n) 
                            + 0.75*s(i,j+2,k,n) 
                            + 0.5 *s(i,j+1,k,n) 
                            - 0.05*s(i,j,k,n);
                    
                    // Make sure sedge lies in between adjacent cell-centered values.
                    sedgerr = max(sedgerr,min(s(i,j+1,k,n),s(i,j+2,k,n)));
                    sedgerr = min(sedgerr,max(s(i,j+1,k,n),s(i,j+2,k,n)));
                }

                // Apply Colella 2008 limiters to compute sm and sp in the second
                // and third inner cells.
                if (j == jhi-1 || j == jhi-2) {

                    alphap = sedger-s(i,j,k,n);
                    alpham = sedge-s(i,j,k,n);
                    bigp = fabs(alphap) > 2.0*fabs(alpham);
                    bigm = fabs(alpham) > 2.0*fabs(alphap);
                    extremum = false;

                    if (alpham*alphap >= 0.0) {
                        extremum = true;
                    } else if (bigp || bigm) {
                        // Possible extremum. We look at cell centered values and face
                        // centered values for a change in sign in the differences adjacent to
                        // the cell. We use the pair of differences whose minimum magnitude is
                        // the largest, and thus least susceptible to sensitivity to roundoff.
                        Real dafacem = sedge - sedgel;
                        Real dafacep = sedgerr - sedger;
                        Real dabarm = s(i,j,k,n) - s(i,j-1,k,n);
                        Real dabarp = s(i,j+1,k,n) - s(i,j,k,n);
                        Real dafacemin = min(fabs(dafacem),fabs(dafacep));
                        Real dabarmin= min(fabs(dabarm),fabs(dabarp));
                        Real dachkm = 0.0;
                        Real dachkp = 0.0;
                        if (dafacemin>=dabarmin) {
                            dachkm = dafacem;
                            dachkp = dafacep;
                        } else {
                            dachkm = dabarm;
                            dachkp = dabarp;
                        }
                        extremum = (dachkm*dachkp <= 0.0);
                    }

                    if (extremum) {
                        Real D2  = 6.0*(alpham + alphap);
                        Real D2L = s(i,j-2,k,n)-2.0*s(i,j-1,k,n)+s(i,j,k,n);
                        Real D2R = s(i,j,k,n)-2.0*s(i,j+1,k,n)+s(i,j+2,k,n);
                        Real D2C = s(i,j-1,k,n)-2.0*s(i,j,k,n)+s(i,j+1,k,n);
                        Real sgn = copysign(1.0,D2);
                        Real D2LIM = max(min(sgn*D2,min(C*sgn*D2L,min(C*sgn*D2R,C*sgn*D2C))),0.0);
                        Real D2ABS = max(fabs(D2),1.e-10);
                        alpham = alpham*D2LIM/D2ABS;
                        alphap = alphap*D2LIM/D2ABS;
                    } else {
                        if (bigp) {
                            Real sgn = copysign(1.0,alpham);
                            Real amax = -alphap*alphap / (4.0*(alpham + alphap));
                            Real delam = s(i,j-1,k,n) - s(i,j,k,n);
                            if (sgn*amax >= sgn*delam) {
                                if (sgn*(delam - alpham) >= 1.e-10) {
                                    alphap = (-2.0*delam - 2.0*sgn*sqrt(delam*delam - delam*alpham));
                                } else {
                                    alphap = -2.0*alpham;
                                }
                            }
                        }
                        if (bigm) {
                            Real sgn = copysign(1.0,alphap);
                            Real amax = -alpham*alpham / (4.0*(alpham + alphap));
                            Real delap = s(i,j+1,k,n) - s(i,j,k,n);
                            if (sgn*amax >= sgn*delap) {
                                if (sgn*(delap - alphap)>=1.e-10) {
                                    alpham = (-2.0*delap - 2.0*sgn*sqrt(delap*delap - delap*alphap));
                                } else {
                                    alpham = -2.0*alphap;
                                }
                            }
                        }
                    }

                    sm = s(i,j,k,n) + alpham;
                    sp = s(i,j,k,n) + alphap;
                }
            }

            ////////////////////////////////////
            // Compute y-component of Ip and Im.
            ////////////////////////////////////
            Real s6 = 6.0*s(i,j,k,n) - 3.0*(sm+sp);

            if (is_umac) {

                // v is MAC velocity -- use edge-based indexing
                Real sigma = fabs(v(i,j+1,k))*dt_local/hy;
                if (v(i,j+1,k) > rel_eps) {
                    Ip(i,j,k,1) = sp - 0.5*sigma*(sp-sm-(1.0-2.0/3.0*sigma)*s6);
                } else {
                    Ip(i,j,k,1) = s(i,j,k,n);
                }

                sigma = fabs(v(i,j,k))*dt_local/hy;
                if (v(i,j,k) < -rel_eps) {
                    Im(i,j,k,1) = sm + 0.5*sigma*(sp-sm+(1.0-2.0/3.0*sigma)*s6);
                } else {
                    Im(i,j,k,1) = s(i,j,k,n);
                }

            } else {

                Real sigma = fabs(v(i,j,k))*dt_local/hy;
                if (v(i,j,k) > rel_eps) {
                    Ip(i,j,k,1) = sp - 0.5*sigma*(sp-sm-(1.0-2.0/3.0*sigma)*s6);
                } else {
                    Ip(i,j,k,1) = s(i,j,k,n);
                }

                sigma = fabs(v(i,j,k))*dt_local/hy;
                if (v(i,j,k) < -rel_eps) {
                    Im(i,j,k,1) = sm + 0.5*sigma*(sp-sm+(1.0-2.0/3.0*sigma)*s6);
                } else {
                    Im(i,j,k,1) = s(i,j,k,n);
                }
            }
        });
    }

#if (AMREX_SPACEDIM == 3)
    /////////////
    // z-dir
    /////////////

    const int klo = domainBox.loVect()[2];
    const int khi = domainBox.hiVect()[2];
    bclo = bcs[bccomp].lo()[2];
    bchi = bcs[bccomp].hi()[2];

    if (ppm_type == 1) {

        AMREX_PARALLEL_FOR_3D(bx, i, j, k, 
        {
            // Compute van Leer slopes in z-direction.

            // sm
            Real dsvl_l = 0.0;
            Real dsvl_r = 0.0;

            // left side
            Real dsc = 0.5 * (s(i,j,k,n) - s(i,j,k-2,n));
            Real dsl = 2.0 * (s(i,j,k-1,n) - s(i,j,k-2,n));
            Real dsr = 2.0 * (s(i,j,k,n) - s(i,j,k-1,n));
            if (dsl*dsr > 0.0) 
                dsvl_l = copysign(1.0,dsc)*min(fabs(dsc),min(fabs(dsl),fabs(dsr)));

            // right side
            dsc = 0.5 * (s(i,j,k+1,n) - s(i,j,k-1,n));
            dsl = 2.0 * (s(i,j,k,n) - s(i,j,k-1,n));
            dsr = 2.0 * (s(i,j,k+1,n) - s(i,j,k,n));
            if (dsl*dsr > 0.0) 
                dsvl_r = copysign(1.0,dsc)*min(fabs(dsc),min(fabs(dsl),fabs(dsr)));

            // Interpolate s to z-edges.
            Real sm = 0.5*(s(i,j,k,n)+s(i,j,k-1,n)) - (dsvl_r-dsvl_l)/6.0;
            
            // Make sure sedge lies in between adjacent cell-centered values.
            sm = max(sm,min(s(i,j,k,n),s(i,j,k-1,n)));
            sm = min(sm,max(s(i,j,k,n),s(i,j,k-1,n)));

            // sp
            dsvl_l = 0.0;
            dsvl_r = 0.0;

            // left side
            dsc = 0.5 * (s(i,j,k+1,n) - s(i,j,k-1,n));
            dsl = 2.0 * (s(i,j,k,n) - s(i,j,k-1,n));
            dsr = 2.0 * (s(i,j,k+1,n) - s(i,j,k,n));
            if (dsl*dsr > 0.0) 
                dsvl_l = copysign(1.0,dsc)*min(fabs(dsc),min(fabs(dsl),fabs(dsr)));

            // right side
            dsc = 0.5 * (s(i,j,k+2,n) - s(i,j,k,n));
            dsl = 2.0 * (s(i,j,k+1,n) - s(i,j,k,n));
            dsr = 2.0 * (s(i,j,k+2,n) - s(i,j,k+1,n));
            if (dsl*dsr > 0.0) 
                dsvl_r = copysign(1.0,dsc)*min(fabs(dsc),min(fabs(dsl),fabs(dsr)));

            // Interpolate s to z-edges.
            Real sp = 0.5*(s(i,j,k+1,n)+s(i,j,k,n)) - (dsvl_r-dsvl_l)/6.0;
            
            // Make sure sedge lies in between adjacent cell-centered values.
            sp = max(sp,min(s(i,j,k+1,n),s(i,j,k,n)));
            sp = min(sp,max(s(i,j,k+1,n),s(i,j,k,n)));

            // save for later 
            Real sedgel = sp;
            Real sedger = sm;

            // Modify using quadratic limiters.
            if ((sp-s(i,j,k,n))*(s(i,j,k,n)-sm) <= 0.0) {
                sp = s(i,j,k,n);
                sm = s(i,j,k,n);
            } else if (fabs(sp-s(i,j,k,n)) >= 2.0*fabs(sm-s(i,j,k,n))) {
                sp = 3.0*s(i,j,k,n) - 2.0*sm;
            } else if (fabs(sm-s(i,j,k,n)) >= 2.0*fabs(sp-s(i,j,k,n))) {
                sm = 3.0*s(i,j,k,n) - 2.0*sp;
            }
            
            // Different stencil needed for z-component of EXT_DIR and HOEXTRAP adv_bc's.
            if (k == klo) {
                if (bclo == EXT_DIR  || bclo == HOEXTRAP) {

                    // The value in the first cc ghost cell represents the edge value.
                    sm = s(i,j,k-1,n);

                    // Use a modified stencil to get sp on the first interior edge.
                    sp = -0.2 *s(i,j,k-1,n) 
                        + 0.75*s(i,j,k,n) 
                        + 0.5 *s(i,j,k+1,n) 
                        - 0.05*s(i,j,k+2,n);
                    
                    // Make sure sp lies in between adjacent cell-centered values.
                    sp = max(sp,min(s(i,j,k+1,n),s(i,j,k,n)));
                    sp = min(sp,max(s(i,j,k+1,n),s(i,j,k,n)));
                }
                    
            } else if (k == klo+1) {
                if (bclo == EXT_DIR  || bclo == HOEXTRAP) {

                    // Use a modified stencil to get sm on the first interior edge.
                    sm = -0.2 *s(i,j,k-2,n) 
                        + 0.75*s(i,j,k-1,n) 
                        + 0.5 *s(i,j,k,n) 
                        - 0.05*s(i,j,k+1,n);
                    
                    // Make sure sm lies in between adjacent cell-centered values.
                    sm = max(sm,min(s(i,j,k,n),s(i,j,k-1,n)));
                    sm = min(sm,max(s(i,j,k,n),s(i,j,k-1,n)));

                    // reset sp on second interior edge
                    sp = sedgel;
                    
                    // Modify using quadratic limiters.
                    if ((sp-s(i,j,k,n))*(s(i,j,k,n)-sm) <= 0.0) {
                        sp = s(i,j,k,n);
                        sm = s(i,j,k,n);
                    } else if (fabs(sp-s(i,j,k,n)) >= 2.0*fabs(sm-s(i,j,k,n))) {
                        sp = 3.0*s(i,j,k,n) - 2.0*sm;
                    } else if (fabs(sm-s(i,j,k,n)) >= 2.0*fabs(sp-s(i,j,k,n))) {
                        sm = 3.0*s(i,j,k,n) - 2.0*sp;
                    }
                }

            } else if (k == khi) {
                if (bchi == EXT_DIR  || bchi == HOEXTRAP) {

                    // The value in the first cc ghost cell represents the edge value.
                    sp = s(i,j,k+1,n);

                    // Use a modified stencil to get sm on the first interior edge.
                    sm = -0.2 *s(i,j,k+1,n) 
                        + 0.75*s(i,j,k,n) 
                        + 0.5 *s(i,j,k-1,n) 
                        - 0.05*s(i,j,k-2,n);
                    
                    // Make sure sm lies in between adjacent cell-centered values.
                    sm = max(sm,min(s(i,j,k-1,n),s(i,j,k,n)));
                    sm = min(sm,max(s(i,j,k-1,n),s(i,j,k,n)));
                }

            } else if (k == khi-1) {
                if (bchi == EXT_DIR  || bchi == HOEXTRAP) {

                    // Use a modified stencil to get sedge on the first interior edge.
                    sp = -0.2 *s(i,j,k+2,n) 
                        + 0.75*s(i,j,k+1,n) 
                        + 0.5 *s(i,j,k,n) 
                        - 0.05*s(i,j,k-1,n);
                    
                    // Make sure sp lies in between adjacent cell-centered values.
                    sp = max(sp,min(s(i,j,k,n),s(i,j,k+1,n)));
                    sp = min(sp,max(s(i,j,k,n),s(i,j,k+1,n)));

                    // reset sm on second interior edge
                    sm = sedger;

                    // Modify using quadratic limiters.
                    if ((sp-s(i,j,k,n))*(s(i,j,k,n)-sm) <= 0.0) {
                        sp = s(i,j,k,n);
                        sm = s(i,j,k,n);
                    } else if (fabs(sp-s(i,j,k,n)) >= 2.0*fabs(sm-s(i,j,k,n))) {
                        sp = 3.0*s(i,j,k,n) - 2.0*sm;
                    } else if (fabs(sm-s(i,j,k,n)) >= 2.0*fabs(sp-s(i,j,k,n))) {
                        sm = 3.0*s(i,j,k,n) - 2.0*sp;
                    }
                }
            }

            ////////////////////////////////////
            // Compute z-component of Ip and Im.
            ////////////////////////////////////
            Real s6 = 6.0*s(i,j,k,n) - 3.0*(sm+sp);

            if (is_umac) {

                // w is MAC velocity -- use edge-based indexing
                Real sigma = fabs(w(i,j,k+1))*dt_local/hz;
                
                if (w(i,j,k+1) > rel_eps) {
                    Ip(i,j,k,2) = sp - 
                        0.5*sigma*(sp-sm-(1.0-2.0/3.0*sigma)*s6);
                } else {
                    Ip(i,j,k,2) = s(i,j,k,n);
                }

                sigma = fabs(w(i,j,k))*dt_local/hz;

                if (w(i,j,k) < -rel_eps) {
                    Im(i,j,k,2) = sm + 
                        0.5*sigma*(sp-sm+(1.0-2.0/3.0*sigma)*s6);
                } else {
                    Im(i,j,k,2) = s(i,j,k,n);
                }
            } else {

                Real sigma = fabs(w(i,j,k))*dt_local/hz;
                
                if (w(i,j,k) > rel_eps) {
                    Ip(i,j,k,2) = sp - 
                        0.5*sigma*(sp-sm-(1.0-2.0/3.0*sigma)*s6);
                } else {
                    Ip(i,j,k,2) = s(i,j,k,n);
                }

                sigma = fabs(w(i,j,k))*dt_local/hz;
                
                if (w(i,j,k) < -rel_eps) {
                    Im(i,j,k,2) = sm + 
                        0.5*sigma*(sp-sm+(1.0-2.0/3.0*sigma)*s6);
                } else {
                    Im(i,j,k,2) = s(i,j,k,n);
                }
            }
        });

    } else if (ppm_type == 2) {

        AMREX_PARALLEL_FOR_3D(bx, i, j, k, 
        {
            // -1
            // Interpolate s to z-edges.
            Real sedgel = (7.0/12.0)*(s(i,j,k-2,n)+s(i,j,k-1,n)) 
                    - (1.0/12.0)*(s(i,j,k-3,n)+s(i,j,k,n));
            
            // Limit sedge.
            if ((sedgel-s(i,j,k-2,n))*(s(i,j,k-1,n)-sedgel) < 0.0) {
                Real D2  = 3.0*(s(i,j,k-2,n)-2.0*sedgel+s(i,j,k-1,n));
                Real D2L = s(i,j,k-3,n)-2.0*s(i,j,k-2,n)+s(i,j,k-1,n);
                Real D2R = s(i,j,k-2,n)-2.0*s(i,j,k-1,n)+s(i,j,k,n);
                Real sgn = copysign(1.0,D2);
                Real D2LIM = sgn*max(min(C*sgn*D2L,min(C*sgn*D2R,sgn*D2)),0.0);
                sedgel = 0.5*(s(i,j,k-2,n)+s(i,j,k-1,n)) - D2LIM/6.0;
            }

            // 0
            // Interpolate s to z-edges.
            Real sedge = (7.0/12.0)*(s(i,j,k-1,n)+s(i,j,k,n)) 
                    - (1.0/12.0)*(s(i,j,k-2,n)+s(i,j,k+1,n));
            
            // Limit sedge.
            if ((sedge-s(i,j,k-1,n))*(s(i,j,k,n)-sedge) < 0.0) {
                Real D2  = 3.0*(s(i,j,k-1,n)-2.0*sedge+s(i,j,k,n));
                Real D2L = s(i,j,k-2,n)-2.0*s(i,j,k-1,n)+s(i,j,k,n);
                Real D2R = s(i,j,k-1,n)-2.0*s(i,j,k,n)+s(i,j,k+1,n);
                Real sgn = copysign(1.0,D2);
                Real D2LIM = sgn*max(min(C*sgn*D2L,min(C*sgn*D2R,sgn*D2)),0.0);
                sedge = 0.5*(s(i,j,k-1,n)+s(i,j,k,n)) - D2LIM/6.0;
            }

            // +1
            // Interpolate s to z-edges.
            Real sedger = (7.0/12.0)*(s(i,j,k,n)+s(i,j,k+1,n)) 
                    - (1.0/12.0)*(s(i,j,k-1,n)+s(i,j,k+2,n));
            
            // Limit sedge.
            if ((sedger-s(i,j,k,n))*(s(i,j,k+1,n)-sedger) < 0.0) {
                Real D2  = 3.0*(s(i,j,k,n)-2.0*sedger+s(i,j,k+1,n));
                Real D2L = s(i,j,k-1,n)-2.0*s(i,j,k,n)+s(i,j,k+1,n);
                Real D2R = s(i,j,k,n)-2.0*s(i,j,k+1,n)+s(i,j,k+2,n);
                Real sgn = copysign(1.0,D2);
                Real D2LIM = sgn*max(min(C*sgn*D2L,min(C*sgn*D2R,sgn*D2)),0.0);
                sedger = 0.5*(s(i,j,k,n)+s(i,j,k+1,n)) - D2LIM/6.0;
            }

            // +2
            // Interpolate s to z-edges.
            Real sedgerr = (7.0/12.0)*(s(i,j,k+1,n)+s(i,j,k+2,n)) 
                    - (1.0/12.0)*(s(i,j,k,n)+s(i,j,k+3,n));
            
            // Limit sedge.
            if ((sedgerr-s(i,j,k+1,n))*(s(i,j,k+2,n)-sedgerr) < 0.0) {
                Real D2  = 3.0*(s(i,j,k+1,n)-2.0*sedgerr+s(i,j,k+2,n));
                Real D2L = s(i,j,k,n)-2.0*s(i,j,k+1,n)+s(i,j,k+2,n);
                Real D2R = s(i,j,k+1,n)-2.0*s(i,j,k+2,n)+s(i,j,k+3,n);
                Real sgn = copysign(1.0,D2);
                Real D2LIM = sgn*max(min(C*sgn*D2L,min(C*sgn*D2R,sgn*D2)),0.0);
                sedgerr = 0.5*(s(i,j,k+1,n)+s(i,j,k+2,n)) - D2LIM/6.0;
            }

            Real alphap = sedger-s(i,j,k,n);
            Real alpham = sedge-s(i,j,k,n);
            bool bigp = fabs(alphap) > 2.0*fabs(alpham);
            bool bigm = fabs(alpham) > 2.0*fabs(alphap);
            bool extremum = false;

            if (alpham*alphap >= 0.0) {
                extremum = true;
            } else if (bigp || bigm) {
                //
                // Possible extremum. We look at cell centered values and face
                // centered values for a change in sign in the differences adjacent to
                // the cell. We use the pair of differences whose minimum magnitude is the
                // largest, and thus least susceptible to sensitivity to roundoff.
                //
                Real dafacem = sedge - sedgel;
                Real dafacep = sedgerr - sedger;
                Real dabarm = s(i,j,k,n) - s(i,j,k-1,n);
                Real dabarp = s(i,j,k+1,n) - s(i,j,k,n);
                Real dafacemin = min(fabs(dafacem),fabs(dafacep));
                Real dabarmin = min(fabs(dabarm),fabs(dabarp));
                Real dachkm = 0.0;
                Real dachkp = 0.0;
                if (dafacemin >= dabarmin) {
                    dachkm = dafacem;
                    dachkp = dafacep;
                } else {
                    dachkm = dabarm;
                    dachkp = dabarp;
                }
                extremum = (dachkm*dachkp <= 0.0);
            }

            if (extremum) {
                Real D2  = 6.0*(alpham + alphap);
                Real D2L = s(i,j,k-2,n)-2.0*s(i,j,k-1,n)+s(i,j,k,n);
                Real D2R = s(i,j,k,n)-2.0*s(i,j,k+1,n)+s(i,j,k+2,n);
                Real D2C = s(i,j,k-1,n)-2.0*s(i,j,k,n)+s(i,j,k+1,n);
                Real sgn = copysign(1.0,D2);
                Real D2LIM = max(min(sgn*D2,min(C*sgn*D2L,min(C*sgn*D2R,C*sgn*D2C))),0.0);
                Real D2ABS = max(fabs(D2),1.e-10);
                alpham = alpham*D2LIM/D2ABS;
                alphap = alphap*D2LIM/D2ABS;
            } else {
                if (bigp) {
                    Real sgn = copysign(1.0,alpham);
                    Real amax = -alphap*alphap / (4.0*(alpham + alphap));
                    Real delam = s(i,j,k-1,n) - s(i,j,k,n);
                    if (sgn*amax >= sgn*delam) {
                        if (sgn*(delam - alpham) >= 1.e-10) {
                            alphap = (-2.0*delam - 2.0*sgn*sqrt(delam*delam - delam*alpham));
                        } else {
                            alphap = -2.0*alpham;
                        }
                    }
                }
                if (bigm) {
                    Real sgn = copysign(1.0,alphap);
                    Real amax = -alpham*alpham / (4.0*(alpham + alphap));
                    Real delap = s(i,j,k+1,n) - s(i,j,k,n);
                    if (sgn*amax >= sgn*delap) {
                        if (sgn*(delap - alphap) >= 1.e-10) {
                            alpham = (-2.0*delap - 2.0*sgn*sqrt(delap*delap - delap*alphap));
                        } else {
                            alpham = -2.0*alphap;
                        }
                    }
                }
            }

            Real sm = s(i,j,k,n) + alpham;
            Real sp = s(i,j,k,n) + alphap;

            // Different stencil needed for z-component of EXT_DIR and HOEXTRAP adv_bc's.
            if (bclo == EXT_DIR  || bclo == HOEXTRAP) {
                if (k == klo) {

                    // The value in the first cc ghost cell represents the edge value.
                    sm = s(i,j,k-1,n);
                    
                    // Use a modified stencil to get sedge on the first interior edge.
                    sp = -0.2 *s(i,j,k-1,n) 
                        + 0.75*s(i,j,k,n) 
                        + 0.5 *s(i,j,k+1,n) 
                        - 0.05*s(i,j,k+2,n);
                    
                    // Make sure sedge lies in between adjacent cell-centered values.
                    sp = max(sp,min(s(i,j,k+1,n),s(i,j,k,n)));
                    sp = min(sp,max(s(i,j,k+1,n),s(i,j,k,n)));

                } else if (k == klo+1) {

                    sedgel = s(i,j,k-2,n);
                    
                    // Use a modified stencil to get sedge on the first interior edge.
                    sedge = -0.2 *s(i,j,k-2,n) 
                           + 0.75*s(i,j,k-1,n) 
                           + 0.5 *s(i,j,k,n) 
                           - 0.05*s(i,j,k+1,n);
                    
                    // Make sure sedge lies in between adjacent cell-centered values.
                    sedge = max(sedge,min(s(i,j,k,n),s(i,j,k-1,n)));
                    sedge = min(sedge,max(s(i,j,k,n),s(i,j,k-1,n)));

                } else if (k == klo+2) {

                    // Use a modified stencil to get sedge on the first interior edge.
                    sedgel = -0.2 *s(i,j,k-3,n) 
                            + 0.75*s(i,j,k-2,n) 
                            + 0.5 *s(i,j,k-1,n) 
                            - 0.05*s(i,j,k,n);
                    
                    // Make sure sedge lies in between adjacent cell-centered values.
                    sedgel = max(sedgel,min(s(i,j,k-1,n),s(i,j,k-2,n)));
                    sedgel = min(sedgel,max(s(i,j,k-1,n),s(i,j,k-2,n)));
                }

                // Apply Colella 2008 limiters to compute sm and sp in the second
                // and third inner cells.
                if (k == klo+1 || k == klo+2) {

                    alphap = sedger-s(i,j,k,n);
                    alpham = sedge-s(i,j,k,n);
                    bigp = fabs(alphap) > 2.0*fabs(alpham);
                    bigm = fabs(alpham) > 2.0*fabs(alphap);
                    extremum = false;

                    if (alpham*alphap >= 0.0) {
                        extremum = true;
                    } else if (bigp || bigm) {
                        // Possible extremum. We look at cell centered values and face
                        // centered values for a change in sign in the differences adjacent to
                        // the cell. We use the pair of differences whose minimum magnitude is
                        // the largest, and thus least susceptible to sensitivity to roundoff.
                        Real dafacem = sedge - sedgel;
                        Real dafacep = sedgerr - sedger;
                        Real dabarm = s(i,j,k,n) - s(i,j,k-1,n);
                        Real dabarp = s(i,j,k+1,n) - s(i,j,k,n);
                        Real dafacemin = min(fabs(dafacem),fabs(dafacep));
                        Real dabarmin= min(fabs(dabarm),fabs(dabarp));
                        Real dachkm = 0.0;
                        Real dachkp = 0.0;
                        if (dafacemin >= dabarmin) {
                            dachkm = dafacem;
                            dachkp = dafacep;
                        } else {
                            dachkm = dabarm;
                            dachkp = dabarp;
                        }
                        extremum = (dachkm*dachkp <= 0.0);
                    }

                    if (extremum) {
                        Real D2  = 6.0*(alpham + alphap);
                        Real D2L = s(i,j,k-2,n)-2.0*s(i,j,k-1,n)+s(i,j,k,n);
                        Real D2R = s(i,j,k,n)-2.0*s(i,j,k+1,n)+s(i,j,k+2,n);
                        Real D2C = s(i,j,k-1,n)-2.0*s(i,j,k,n)+s(i,j,k+1,n);
                        Real sgn = copysign(1.0,D2);
                        Real D2LIM = max(min(sgn*D2,min(C*sgn*D2L,min(C*sgn*D2R,C*sgn*D2C))),0.0);
                        Real D2ABS = max(fabs(D2),1.e-10);
                        alpham = alpham*D2LIM/D2ABS;
                        alphap = alphap*D2LIM/D2ABS;
                    } else {
                        if (bigp) {
                            Real sgn = copysign(1.0,alpham);
                            Real amax = -alphap*alphap / (4.0*(alpham + alphap));
                            Real delam = s(i,j,k-1,n) - s(i,j,k,n);
                            if (sgn*amax >= sgn*delam) {
                                if (sgn*(delam - alpham) >= 1.e-10) {
                                    alphap = (-2.0*delam - 2.0*sgn*sqrt(delam*delam - delam*alpham));
                                } else {
                                    alphap = -2.0*alpham;
                                }
                            }
                        }
                        if (bigm) {
                            Real sgn = copysign(1.0,alphap);
                            Real amax = -alpham*alpham / (4.0*(alpham + alphap));
                            Real delap = s(i,j,k+1,n) - s(i,j,k,n);
                            if (sgn*amax >= sgn*delap) {
                                if (sgn*(delap - alphap) >= 1.e-10) {
                                    alpham = (-2.0*delap - 2.0*sgn*sqrt(delap*delap - delap*alphap));
                                } else {
                                    alpham = -2.0*alphap;
                                }
                            }
                        }
                    }

                    sm = s(i,j,k,n) + alpham;
                    sp = s(i,j,k,n) + alphap;
                }
            }

            if (bchi == EXT_DIR  || bchi == HOEXTRAP) {
                if (k == khi) {

                    // The value in the first cc ghost cell represents the edge value.
                    sp = s(i,j,k+1,n);
                    
                    // Use a modified stencil to get sedge on the first interior edge.
                    sm = -0.2 *s(i,j,k+1,n) 
                        + 0.75*s(i,j,k,n) 
                        + 0.5 *s(i,j,k-1,n) 
                        - 0.05*s(i,j,k-2,n);
                    
                    // Make sure sedge lies in between adjacent cell-centered values.
                    sm = max(sm,min(s(i,j,k-1,n),s(i,j,k,n)));
                    sm = min(sm,max(s(i,j,k-1,n),s(i,j,k,n)));

                } else if (k == khi-1) {

                    sedgerr = s(i,j,k+2,n);
                    
                    // Use a modified stencil to get sedge on the first interior edge.
                    sedger = -0.2 *s(i,j,k+2,n) 
                            + 0.75*s(i,j,k+1,n) 
                            + 0.5 *s(i,j,k,n) 
                            - 0.05*s(i,j,k-1,n);
                    
                    // Make sure sedge lies in between adjacent cell-centered values.
                    sedger = max(sedger,min(s(i,j,k,n),s(i,j,k+1,n)));
                    sedger = min(sedger,max(s(i,j,k,n),s(i,j,k+1,n)));

                } else if (k == khi-2) {

                    // Use a modified stencil to get sedge on the first interior edge.
                    sedgerr = -0.2 *s(i,j,k+3,n) 
                             + 0.75*s(i,j,k+2,n) 
                             + 0.5 *s(i,j,k+1,n) 
                             - 0.05*s(i,j,k,n);
                    
                    // Make sure sedge lies in between adjacent cell-centered values.
                    sedgerr = max(sedgerr,min(s(i,j,k+1,n),s(i,j,k+2,n)));
                    sedgerr = min(sedgerr,max(s(i,j,k+1,n),s(i,j,k+2,n)));
                }

                // Apply Colella 2008 limiters to compute sm and sp in the second
                // and third inner cells.
                if (k == khi-1 || k == khi-2) {

                    alphap = sedger-s(i,j,k,n);
                    alpham = sedge-s(i,j,k,n);
                    bigp = fabs(alphap) > 2.0*fabs(alpham);
                    bigm = fabs(alpham) > 2.0*fabs(alphap);
                    extremum = false;

                    if (alpham*alphap >= 0.0) {
                        extremum = true;
                    } else if (bigp || bigm) {
                        // Possible extremum. We look at cell centered values and face
                        // centered values for a change in sign in the differences adjacent to
                        // the cell. We use the pair of differences whose minimum magnitude is
                        // the largest, and thus least susceptible to sensitivity to roundoff.
                        Real dafacem = sedge - sedgel;
                        Real dafacep = sedgerr - sedger;
                        Real dabarm = s(i,j,k,n) - s(i,j,k-1,n);
                        Real dabarp = s(i,j,k+1,n) - s(i,j,k,n);
                        Real dafacemin = min(fabs(dafacem),fabs(dafacep));
                        Real dabarmin= min(fabs(dabarm),fabs(dabarp));
                        Real dachkm = 0.0;
                        Real dachkp = 0.0;
                        if (dafacemin >= dabarmin) {
                            dachkm = dafacem;
                            dachkp = dafacep;
                        } else {
                            dachkm = dabarm;
                            dachkp = dabarp;
                        }
                        extremum = (dachkm*dachkp <= 0.0);
                    }

                    if (extremum) {
                        Real D2  = 6.0*(alpham + alphap);
                        Real D2L = s(i,j,k-2,n)-2.0*s(i,j,k-1,n)+s(i,j,k,n);
                        Real D2R = s(i,j,k,n)-2.0*s(i,j,k+1,n)+s(i,j,k+2,n);
                        Real D2C = s(i,j,k-1,n)-2.0*s(i,j,k,n)+s(i,j,k+1,n);
                        Real sgn = copysign(1.0,D2);
                        Real D2LIM = max(min(sgn*D2,min(C*sgn*D2L,min(C*sgn*D2R,C*sgn*D2C))),0.0);
                        Real D2ABS = max(fabs(D2),1.e-10);
                        alpham = alpham*D2LIM/D2ABS;
                        alphap = alphap*D2LIM/D2ABS;
                    } else {
                        if (bigp) {
                            Real sgn = copysign(1.0,alpham);
                            Real amax = -alphap*alphap / (4.0*(alpham + alphap));
                            Real delam = s(i,j,k-1,n) - s(i,j,k,n);
                            if (sgn*amax >= sgn*delam) {
                                if (sgn*(delam - alpham) >= 1.e-10) {
                                    alphap = (-2.0*delam - 2.0*sgn*sqrt(delam*delam - delam*alpham));
                                } else {
                                    alphap = -2.0*alpham;
                                }
                            }
                        }
                        if (bigm) {
                            Real sgn = copysign(1.0,alphap);
                            Real amax = -alpham*alpham / (4.0*(alpham + alphap));
                            Real delap = s(i,j,k+1,n) - s(i,j,k,n);
                            if (sgn*amax >= sgn*delap) {
                                if (sgn*(delap - alphap) >= 1.e-10) {
                                    alpham = (-2.0*delap - 2.0*sgn*sqrt(delap*delap - delap*alphap));
                                } else {
                                    alpham = -2.0*alphap;
                                }
                            }
                        }
                    }

                    sm = s(i,j,k,n) + alpham;
                    sp = s(i,j,k,n) + alphap;
                }
            }

            ////////////////////////////////////
            // Compute z-component of Ip and Im.
            ////////////////////////////////////

            Real s6 = 6.0*s(i,j,k,n) - 3.0*(sm+sp);

            if (is_umac) {

                // w is MAC velocity -- use edge-based indexing
                Real sigma = fabs(w(i,j,k+1))*dt_local/hz;
                
                if (w(i,j,k+1) > rel_eps) {
                    Ip(i,j,k,2) = sp - 
                        0.5*sigma*(sp-sm-(1.0-2.0/3.0*sigma)*s6);
                } else {
                    Ip(i,j,k,2) = s(i,j,k,n);
                }

                sigma = fabs(w(i,j,k))*dt_local/hz;
                
                if (w(i,j,k) < -rel_eps) {
                    Im(i,j,k,2) = sm + 
                        0.5*sigma*(sp-sm+(1.0-2.0/3.0*sigma)*s6);
                } else {
                    Im(i,j,k,2) = s(i,j,k,n);
                }
            } else {
                Real sigma = fabs(w(i,j,k))*dt_local/hz;
                
                if (w(i,j,k) > rel_eps) {
                    Ip(i,j,k,2) = sp - 
                        0.5*sigma*(sp-sm-(1.0-2.0/3.0*sigma)*s6);
                } else {
                    Ip(i,j,k,2) = s(i,j,k,n);
                }

                sigma = fabs(w(i,j,k))*dt_local/hz;
                
                if (w(i,j,k) < -rel_eps) {
                    Im(i,j,k,2) = sm + 
                        0.5*sigma*(sp-sm+(1.0-2.0/3.0*sigma)*s6);
                } else {
                    Im(i,j,k,2) = s(i,j,k,n);
                }
            }
        });  
    }
#endif
}