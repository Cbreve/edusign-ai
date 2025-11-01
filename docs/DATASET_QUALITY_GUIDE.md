# Dataset Quality Assurance Guide

This guide explains how to ensure your GSL sign language dataset maintains high quality.

## Quality Metrics

A high-quality dataset should have:

1. **Image Quality**
   - Minimum resolution: 200x200 pixels (40,000 total pixels)
   - Good contrast and clarity
   - Consistent formats (preferably JPEG)
   - No corruption or artifacts

2. **Data Integrity**
   - 95%+ dictionary coverage (entries linked to images)
   - No orphaned images (all images linked to dictionary entries)
   - No missing image references

3. **Dictionary Completeness**
   - All entries have: sign name, meaning, page number
   - Unique IDs for all entries
   - Page numbers within valid range (1-292)

4. **Naming Consistency**
   - Files follow pattern: `page_{number}_img_{number}.{ext}`
   - Consistent file extensions

## Tools Available

### 1. Quality Assurance Check
```bash
python3 scripts/dataset_quality_assurance.py
```
- Checks all quality metrics
- Generates comprehensive report
- Provides quality score (0-100)
- Identifies issues and recommendations

**Output**: `backend/app/data/processed/qa_report.json`

### 2. Automatic Issue Fixing
```bash
# Link orphaned images to dictionary entries
python3 scripts/fix_dataset_issues.py --link-orphaned

# Enhance low-quality images (contrast, sharpness)
python3 scripts/fix_dataset_issues.py --enhance

# Run all fixes
python3 scripts/fix_dataset_issues.py --all
```

### 3. Image Validation
```bash
# Validate extracted images
python3 scripts/validate_sign_images.py --report-only
```

### 4. Sync Dictionary with Images
```bash
# Remove references to deleted images
python3 scripts/sync_dictionary_with_images.py
```

## Quality Workflow

### Initial Dataset Creation
1. Extract images: `python3 scripts/extract_gsl_sign_images.py`
2. Validate images: `python3 scripts/validate_sign_images.py --report-only`
3. Review and remove non-sign images manually
4. Sync dictionary: `python3 scripts/sync_dictionary_with_images.py`
5. Run QA: `python3 scripts/dataset_quality_assurance.py`
6. Fix issues: `python3 scripts/fix_dataset_issues.py --all`
7. Re-run QA to verify improvements

### Ongoing Maintenance
1. Run QA regularly after changes
2. Fix issues automatically where possible
3. Review recommendations and address manually
4. Keep quality score above 80

## Quality Score Breakdown

- **90-100**: Excellent - Ready for production
- **75-89**: Good - Minor issues to address
- **60-74**: Fair - Needs improvement
- **40-59**: Poor - Significant issues
- **0-39**: Critical - Major problems

## Common Issues and Solutions

### Low Resolution Images
**Issue**: Images below 200x200px lack detail
**Solution**: Re-extract from PDF at higher DPI

### Orphaned Images
**Issue**: Images not linked to dictionary entries
**Solution**: `python3 scripts/fix_dataset_issues.py --link-orphaned`

### Low Contrast Images
**Issue**: Images are blurry or hard to see
**Solution**: `python3 scripts/fix_dataset_issues.py --enhance`

### Missing Dictionary Links
**Issue**: Dictionary entries without image references
**Solution**: Re-extract images or manually link

### Inconsistent Naming
**Issue**: Files don't follow naming convention
**Solution**: Rename files to match `page_{number}_img_{number}.{ext}` pattern

## Best Practices

1. **Regular QA Checks**: Run QA after any dataset changes
2. **Version Control**: Track dataset versions and QA scores
3. **Documentation**: Document any manual interventions
4. **Validation**: Always validate before model training
5. **Backup**: Keep backups before making bulk changes

## Target Quality Metrics

For production-ready dataset:
- Quality Score: ≥ 85
- Image Coverage: ≥ 95%
- Image Quality: All images ≥ 200x200px
- Dictionary Completeness: 100% required fields
- No Critical Issues: 0 critical issues

## Files Generated

- `qa_report.json` - Comprehensive quality report
- `image_validation_report.json` - Image validation details
- Quality reports stored in: `backend/app/data/processed/`

