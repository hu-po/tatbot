# A page pose, published by tatbot_bridge on /tatbot/page. The page frame is ros/README.md section 3's
# page/<pattern_id>: origin at the page centre, x along stencil u (right), y toward the top of the print (-v),
# z out of the paper. The bridge also broadcasts it as the tf frame page/<pattern_id>.

uint8 SOURCE_MEASURED=1   # stencild measured it this turn
uint8 SOURCE_LOST=2       # stencild lost it; pose is the last measured one (the last pose stands)
uint8 SOURCE_FIXED=3      # page_source fixed: the nominal pose from stack.yaml (mock, bench fallback)

string pattern_id         # "stencil-<sha256>"; "fixed_<arm>" for SOURCE_FIXED
string print_id           # decoded physical print-instance id; "" if unknown
bool identity_verified
geometry_msgs/PoseWithCovarianceStamped pose  # parent_from_page; header.frame_id "world" (stencil) or "<arm>/base_link" (fixed);
                                              # header.stamp = capture time; covariance row-major (x y z rx ry rz), m^2 and rad^2
float64 translation_sigma_m
float64 rotation_sigma_rad
uint8 source              # SOURCE_*
string support_json       # stencild's support object, verbatim JSON ("" for fixed)
