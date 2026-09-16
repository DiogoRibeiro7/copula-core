//! Internal macros.

/// Implements `Serialize`, `Deserialize`, and `SerializableCopula` for a copula
/// whose state is exactly its constructor parameters.
///
/// Serialization writes the named fields. Deserialization reads them and passes
/// them through the given constructor expression, so parameters that the
/// constructor rejects produce a deserialization error instead of an invalid
/// copula. Unknown fields are rejected.
///
/// The expansion is empty unless the `serde` feature is enabled.
macro_rules! validated_serde {
    ($name:literal, $ty:ident { $($field:ident: $field_ty:ty),+ $(,)? } => $construct:expr) => {
        #[cfg(feature = "serde")]
        const _: () = {
            #[derive(::serde::Serialize)]
            #[serde(rename = $name)]
            struct Repr<'a> {
                $($field: &'a $field_ty,)+
            }

            #[derive(::serde::Deserialize)]
            #[serde(rename = $name, deny_unknown_fields)]
            struct Owned {
                $($field: $field_ty,)+
            }

            impl ::serde::Serialize for $ty {
                fn serialize<S: ::serde::Serializer>(
                    &self,
                    serializer: S,
                ) -> ::core::result::Result<S::Ok, S::Error> {
                    ::serde::Serialize::serialize(&Repr { $($field: &self.$field,)+ }, serializer)
                }
            }

            impl<'de> ::serde::Deserialize<'de> for $ty {
                fn deserialize<D: ::serde::Deserializer<'de>>(
                    deserializer: D,
                ) -> ::core::result::Result<Self, D::Error> {
                    let Owned { $($field,)+ } = <Owned as ::serde::Deserialize>::deserialize(deserializer)?;
                    ($construct).map_err(::serde::de::Error::custom)
                }
            }

            impl $crate::traits::SerializableCopula for $ty {}
        };
    };
}
