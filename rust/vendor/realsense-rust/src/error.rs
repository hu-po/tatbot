//! Defines utilities for dealing with errors across the crate

/// Helper macro for checking errors that are returned from the low-level C-API.
///
/// # Why a macro?
///
/// Understanding why we use a macro instead of e.g. a type is relevant in understanding what the
/// macro does (and vice-versa). First, consider that while the C-API returns a single type for
/// every possible error that can occur (specifically, `*mut rs2_error`), we may not want to
/// describe every possible error across the high-level bindings in this way. This would be
/// problematic because:
///
/// * Some errors do not come from the FFI / C-API
/// * Having more specific types for our errors means that users don't have to check every possible
///   error state. It makes errors more actionable, and means that users can think about individual
///   functions / APIs in terms of the limited scope of errors that are possible.
/// * We should not return raw pointers to users, and try as hard as we can to limit their scope to
///   unsafe blocks!!!
///
/// This macro then is a way to take in two items:
///
/// 1. The `*mut rs2_error`
/// 2. An expression that will evaluate to an tuple-like error that holds `(Rs2Exception, String)`.
///
/// The latter part may seem confusing, but more or less we aim for:
///
/// ```no_run
/// use realsense_rust::kind::Rs2Exception;
///
/// pub enum MyError {
///     // This can be used with the macro
///     FooOccurred(Rs2Exception, String),
///     // This cannot
///     BarDidHappen,
///     // This also **cannot** be used!
///     BazBarBongo {
///         exception: Rs2Exception,
///         message: String,
///     },
/// }
///
/// // This can be used with the macro as well
/// pub struct YourError(pub Rs2Exception, pub String);
/// ```
///
/// Why this form? Every `*mut rs2_error` will provide us with some information per the underlying
/// API. If you look at bindings.rs you'll see what information can be extracted from an
/// `rs2_error` type:
///
/// * `rs2_get_librealsense_exception_type`
/// * `rs2_get_failed_function`
/// * `rs2_get_failed_args`
/// * `rs2_get_error_message`
///
/// This is more or less the information unwound from a C++ exception. librealsense2 adds
/// "exception type" as an enumeration to the mix compared to normal C++ exceptions, but outside of
/// that there is not much remarkable going on here.
///
/// ## Why don't errors include function / arg information?
///
/// Based on the above API, you might think that errors should incorporate `(Rs2Exception, String,
/// String, String)`. This would give us the full information unwound from the original C++
/// exception that underpins librealsense2. In practice, however, this information isn't terribly
/// useful.
///
/// Most of the Rust API sticks to only calling a single librealsense2 function from the FFI at a
/// time. In cases where more than one FFI function is called, errors are categorized into
/// different enum variants on the Rust side to be more specific. So the error variant / type name
/// should tell you what the "failed function" is (and the type is more useful than a string!).
///
/// In the case of "failed args," we do our best to try and scope the types of our inputs into the
/// Rust API so that this doesn't happen. In some cases (such as with
/// [`get_option`](crate::sensor::Sensor::get_option`)) this may not be possible, but we opt
/// instead to return `Option` in such cases, since the failure is expected. Mostly, the "failed
/// args" information isn't actionable by users, and so adding it to the API and increasing the
/// burden on our users (and the readability of the final code) outweighs any benefits we could get
/// from it. If you find yourself passing in arguments often that don't make sense and result in
/// generalized errors, this is probably a failure of our API and something we should fix at the
/// type level. This way, users have a better chance of doing the right thing rather than just
/// getting more informative error messages.
///
/// # How does the macro work?
///
/// It expands to a scoped block-expression that:
///
/// 1. Type checks the error (that way the first argument has to be a `*mut rs2_error`)
/// 2. Performs a null check on that error
/// 3. Converts that null check to either an `Ok(())` or constructs your custom error type if the
///    pointer is non-null.
///
/// Having this expand to a block-expression has some benefits. The chief one is that when you use
/// this macro the expression returns, which means that if you fail to check the result the
/// compiler will warn you.
///
/// ```ignore
/// check_rs2_error!(err, MyError::FooOccurred);
/// ```
///
/// Notice how that expression doesn't end in a `?` character? The compiler will shout at you. In
/// comparison, if we used a type (as was done in 0.5 and earlier releases), we would have to defer
/// that failure to runtime. While the output of the above would be a warning (not a compiler
/// failure), we do enforce that clippy gets run before we merge anything so that such a warning
/// will fail CI, as well as alert users who run it locally that they have made a mistake.
///
/// The other benefit of using a block-expression here is that it integrates well with the rest of
/// standard Rust when writing code. The `?` operator brought up above is one such way, but one can
/// also integrate this into `match` statements, `if let Ok(_) = ...`, etc. These are less useful
/// than the compiler warning, but imposes fewer restrictions on how one writes code.
///
/// ## What if someone just... doesn't use the macro?
///
/// Can't help you there. You can always write code that _doesn't_ use the abstractions you're
/// offered.
///
/// # Examples
///
/// ```ignore
/// use realsense_sys as sys;
///
/// pub struct MyError(pub Rs2Exception, String);
///
/// unsafe {
///     let mut err = std::ptr::null_mut::<sys::rs2_error>();
///     sys::rs2_create_context(
///         -1,
///         &mut err,
///     );
///
///     check_rs2_error!(err, MyError)?;
/// }
///
/// ```
///
#[doc(hidden)]
#[macro_export]
macro_rules! check_rs2_error {
    ($rs2_error:expr, $result:expr) => {
        // We make this alias here to type check $rs2_error.
        {
            use num_traits::FromPrimitive;
            use std::convert::TryInto;
            use $crate::kind::Rs2Exception;
            let err: *mut sys::rs2_error = $rs2_error;
            if err.as_ref().is_some() {
                let realsense_exception_type = sys::rs2_get_librealsense_exception_type(err);
                let realsense_exception_type_i32 = realsense_exception_type.try_into().unwrap();

                let res = $result(
                    Rs2Exception::from_i32(realsense_exception_type_i32).unwrap_or_else(|| {
                        panic!("Unknown Rs2Exception: {}", realsense_exception_type_i32)
                    }),
                    std::ffi::CStr::from_ptr(sys::rs2_get_error_message(err))
                        .to_str()
                        .unwrap()
                        .to_string(),
                );
                sys::rs2_free_error(err);
                Err(res)
            } else {
                Ok(())
            }
        }
    };
}
