from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class Passport(BaseModel):
    full_name: Optional[str] = None
    given_name: Optional[str] = None
    surname: Optional[str] = None
    dob: Optional[str] = None
    country: Optional[str] = None
    sex: Optional[str] = None
    number: Optional[str] = None
    issue_country: Optional[str] = None
    expiry_date: Optional[str] = None


class G28Client(BaseModel):
    full_name: Optional[str] = None
    given_name: Optional[str] = None
    surname: Optional[str] = None
    dob: Optional[str] = None
    country: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None


class G28Attorney(BaseModel):
    name: Optional[str] = None
    firm: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None


class G28(BaseModel):
    client: G28Client = Field(default_factory=G28Client)
    attorney: G28Attorney = Field(default_factory=G28Attorney)


class ValidationResult(BaseModel):
    passport_matches_client: Optional[bool] = None
    issues: List[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    passport: Passport = Field(default_factory=Passport)
    g28: G28 = Field(default_factory=G28)
    summary: Optional[str] = None
    validation: ValidationResult = Field(default_factory=ValidationResult)
    source: Dict[str, str] = Field(default_factory=dict)
    missing_fields: List[str] = Field(default_factory=list)

    def compute_missing(self) -> None:
        missing: List[str] = []
        for path, value in flatten_result(self).items():
            if value in (None, "", []):
                missing.append(path)
        self.missing_fields = missing


def flatten_result(result: ExtractionResult) -> Dict[str, Optional[str]]:
    return {
        "passport.full_name": result.passport.full_name,
        "passport.given_name": result.passport.given_name,
        "passport.surname": result.passport.surname,
        "passport.dob": result.passport.dob,
        "passport.country": result.passport.country,
        "passport.sex": result.passport.sex,
        "passport.number": result.passport.number,
        "passport.issue_country": result.passport.issue_country,
        "passport.expiry_date": result.passport.expiry_date,
        "g28.client.full_name": result.g28.client.full_name,
        "g28.client.given_name": result.g28.client.given_name,
        "g28.client.surname": result.g28.client.surname,
        "g28.client.dob": result.g28.client.dob,
        "g28.client.country": result.g28.client.country,
        "g28.client.address": result.g28.client.address,
        "g28.client.city": result.g28.client.city,
        "g28.client.state": result.g28.client.state,
        "g28.client.zip": result.g28.client.zip,
        "g28.client.email": result.g28.client.email,
        "g28.client.phone": result.g28.client.phone,
        "g28.attorney.name": result.g28.attorney.name,
        "g28.attorney.firm": result.g28.attorney.firm,
        "g28.attorney.address": result.g28.attorney.address,
        "g28.attorney.city": result.g28.attorney.city,
        "g28.attorney.state": result.g28.attorney.state,
        "g28.attorney.zip": result.g28.attorney.zip,
        "g28.attorney.email": result.g28.attorney.email,
        "g28.attorney.phone": result.g28.attorney.phone,
    }
