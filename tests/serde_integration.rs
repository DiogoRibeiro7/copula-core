//! Serialization round trips and validation on deserialization.
#![cfg(feature = "serde")]

use copula_core::prelude::*;
use copula_core::traits::SerializableCopula;
use copula_core::{EmpiricalCopula, MarshallOlkinCopula};
use nalgebra::DMatrix;

/// Serialize, deserialize, and serialize again; the JSON must be identical.
fn round_trip<C: SerializableCopula>(copula: &C) -> C {
    let json = copula.to_json().unwrap();
    let restored = C::from_json(&json).unwrap();
    assert_eq!(restored.to_json().unwrap(), json);
    restored
}

fn assert_same_cdf(a: &impl Copula, b: &impl Copula) {
    for point in [[0.2, 0.7], [0.5, 0.5], [0.9, 0.1]] {
        assert_eq!(a.cdf(&point).unwrap(), b.cdf(&point).unwrap());
    }
}

#[test]
fn archimedean_copulas_round_trip() {
    let clayton = ClaytonCopula::new(2.5).unwrap();
    assert_same_cdf(&clayton, &round_trip(&clayton));
    let gumbel = GumbelCopula::new(1.7).unwrap();
    assert_same_cdf(&gumbel, &round_trip(&gumbel));
    let frank = FrankCopula::new(-3.0).unwrap();
    assert_same_cdf(&frank, &round_trip(&frank));
    let joe = JoeCopula::new(2.2).unwrap();
    assert_same_cdf(&joe, &round_trip(&joe));
    let amh = AMHCopula::new(-0.4).unwrap();
    assert_same_cdf(&amh, &round_trip(&amh));
}

#[test]
fn elliptical_copulas_round_trip() {
    let corr = DMatrix::from_row_slice(3, 3, &[1.0, 0.3, -0.2, 0.3, 1.0, 0.5, -0.2, 0.5, 1.0]);

    let gaussian = GaussianCopula::new(corr.clone()).unwrap();
    let restored = round_trip(&gaussian);
    let point = [0.2, 0.6, 0.8];
    assert_eq!(gaussian.pdf(&point).unwrap(), restored.pdf(&point).unwrap());

    // The Student-t CDF is a Monte Carlo estimate, so compare densities.
    let student_t = StudentTCopula::new(corr, 4.5).unwrap();
    let restored = round_trip(&student_t);
    assert_eq!(
        student_t.pdf(&point).unwrap(),
        restored.pdf(&point).unwrap()
    );
}

#[test]
fn other_copulas_round_trip() {
    let marshall_olkin = MarshallOlkinCopula::new(0.3, 0.6).unwrap();
    assert_same_cdf(&marshall_olkin, &round_trip(&marshall_olkin));

    let data = DMatrix::from_row_slice(4, 2, &[0.1, 0.4, 0.3, 0.2, 0.6, 0.9, 0.8, 0.7]);
    let empirical = EmpiricalCopula::new(data).unwrap();
    assert_same_cdf(&empirical, &round_trip(&empirical));
}

#[test]
fn serialized_form_is_the_parameters() {
    assert_eq!(
        ClaytonCopula::new(2.0).unwrap().to_json().unwrap(),
        r#"{"theta":2.0}"#
    );
    assert_eq!(
        MarshallOlkinCopula::new(0.25, 0.5)
            .unwrap()
            .to_json()
            .unwrap(),
        r#"{"alpha":0.25,"beta":0.5}"#
    );
}

#[test]
fn deserialization_rejects_invalid_parameters() {
    let invalid_theta = ClaytonCopula::from_json(r#"{"theta":-1.0}"#);
    assert!(matches!(
        invalid_theta,
        Err(CopulaError::SerializationError { .. })
    ));
    assert!(GumbelCopula::from_json(r#"{"theta":0.5}"#).is_err());
    assert!(AMHCopula::from_json(r#"{"theta":1.5}"#).is_err());
    assert!(MarshallOlkinCopula::from_json(r#"{"alpha":0.2,"beta":1.0}"#).is_err());

    // A correlation outside [-1, 1].
    let corr = DMatrix::from_row_slice(2, 2, &[1.0, 0.5, 0.5, 1.0]);
    let json = GaussianCopula::new(corr).unwrap().to_json().unwrap();
    assert!(GaussianCopula::from_json(&json.replace("0.5", "1.5")).is_err());

    // Degrees of freedom must be positive.
    let json = StudentTCopula::new_identity(2, 4.0)
        .unwrap()
        .to_json()
        .unwrap();
    assert!(StudentTCopula::from_json(&json.replace("4.0", "-4.0")).is_err());
}

#[test]
fn deserialization_rejects_unknown_and_missing_fields() {
    assert!(ClaytonCopula::from_json(r#"{"theta":2.0,"extra":1}"#).is_err());
    assert!(ClaytonCopula::from_json("{}").is_err());
    assert!(MarshallOlkinCopula::from_json(r#"{"alpha":0.2}"#).is_err());
}
