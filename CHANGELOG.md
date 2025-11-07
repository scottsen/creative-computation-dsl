# Changelog

All notable changes to Kairo will be documented in this file.

## [0.3.1] - 2025-11-07

### Added - Struct Literals & Complete v0.3.1 Feature Set

#### Struct Literal Support
- **Struct instantiation syntax**: `Point { x: 3.0, y: 4.0 }`
- **Field access**: `particle.position`, `particle.velocity`
- **Nested structs**: Full support for structs containing other structs
- **Computed field values**: Field values can be any expression including field access and operations
- **Comprehensive validation**: Clear error messages for missing fields, invalid fields, and undefined types

#### Complete v0.3.1 Features
- Function definitions with typed/untyped parameters
- Lambda expressions with full closure support
- If/else expressions returning values
- Enhanced flow blocks with dt, steps, and substeps parameters
- Struct definitions and literals (NEW)
- Return statements with early exit
- Recursion support
- Higher-order functions
- Physical unit type annotations

#### Testing
- **Parser tests**: 19/19 passing (100%)
- **Runtime tests**: 41/41 passing (100%)
- **Total v0.3.1 tests**: 60 passing, 1 skipped
- Added 12 comprehensive struct literal tests
- All edge cases covered with clear error validation

#### Examples
- `v0_3_1_struct_physics.kairo`: Comprehensive physics simulation demonstrating:
  - Nested struct definitions (Vector2D, Particle)
  - Struct manipulation in functions
  - Field access and updates
  - Real physics with gravity and bouncing

#### Code Quality
- Production-ready implementation
- Comprehensive error messages with context
- Full docstrings on all new methods
- Clean, maintainable code following project patterns

### Changed
- Parser now correctly disambiguates struct literals from if/else block syntax
- Struct literal parsing handles multi-line field lists with proper newline handling

### Fixed
- Parser infinite loop issue with struct literals containing newlines
- Field access now properly works with StructInstance through `__getattr__`

## Previous Versions

### [0.3.0] - 2025-11
- Function definitions
- Lambda expressions
- If/else expressions
- Enhanced flow blocks
- Struct definitions (type system only)

### [0.2.x] - Previous
- Basic temporal computation
- Field operations
- Simple flow blocks
