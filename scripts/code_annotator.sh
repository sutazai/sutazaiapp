#!/bin/bash

# Automatic code annotation
annotate_code() {
    local file=$1
    
    # Add warning comments for duplicates
    sed -i '/#DUPLICATE:/s/$/ #WARNING: Duplicate code detected!/' "$file"
    
    # Add warning comments for similar code
    sed -i '/#SIMILAR:/s/$/ #WARNING: Similar code pattern!/' "$file"
    
    # Add security annotations
    sed -i '/password =/s/$/ #SECURITY: Potential credential exposure!/' "$file"
} 