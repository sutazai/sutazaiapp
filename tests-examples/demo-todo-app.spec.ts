import { test, expect, type Page } from '@playwright/test';

test.beforeEach(async ({ page }) => {
});

  'buy some cheese',
  'feed the cat',
  'book a doctors appointment'
] as const;



    ]);


    ]);

  });

  test('should clear text input field when an item is added', async ({ page }) => {


    // Check that input is empty.
  });

  test('should append new items to the bottom of the list', async ({ page }) => {
    // Create 3 items.

  
    // Check test using different methods.
    await expect(page.getByText('3 items left')).toBeVisible();

    // Check all items in one call.
  });
});

test.describe('Mark all as completed', () => {
  test.beforeEach(async ({ page }) => {
  });

  test.afterEach(async ({ page }) => {
  });

  test('should allow me to mark all items as completed', async ({ page }) => {
    await page.getByLabel('Mark all as complete').check();

  });

  test('should allow me to clear the complete state of all items', async ({ page }) => {
    const toggleAll = page.getByLabel('Mark all as complete');
    // Check and then immediately uncheck.
    await toggleAll.check();
    await toggleAll.uncheck();

    // Should be no completed classes.
  });

  test('complete all checkbox should update state when items are completed / cleared', async ({ page }) => {
    const toggleAll = page.getByLabel('Mark all as complete');
    await toggleAll.check();
    await expect(toggleAll).toBeChecked();


    // Reuse toggleAll locator and make sure its not checked.
    await expect(toggleAll).not.toBeChecked();


    // Assert the toggle all is checked again.
    await expect(toggleAll).toBeChecked();
  });
});

test.describe('Item', () => {

  test('should allow me to mark items as complete', async ({ page }) => {

    // Create two items.
    }

    // Check first item.

    // Check second item.

    // Assert completed class.
  });

  test('should allow me to un-mark items as complete', async ({ page }) => {

    // Create two items.
    }



  });

  test('should allow me to edit an item', async ({ page }) => {


    // Explicitly assert the new text value.
      'buy some sausages',
    ]);
  });
});

test.describe('Editing', () => {
  test.beforeEach(async ({ page }) => {
  });

  test('should hide other controls when editing', async ({ page }) => {
    })).not.toBeVisible();
  });

  test('should save edits on blur', async ({ page }) => {

      'buy some sausages',
    ]);
  });

  test('should trim entered text', async ({ page }) => {

      'buy some sausages',
    ]);
  });

  test('should remove the item if an empty text string was entered', async ({ page }) => {

    ]);
  });

  test('should cancel edits on escape', async ({ page }) => {
  });
});

test.describe('Counter', () => {
    




  });
});

test.describe('Clear completed button', () => {
  test.beforeEach(async ({ page }) => {
  });

  test('should display the correct text', async ({ page }) => {
    await expect(page.getByRole('button', { name: 'Clear completed' })).toBeVisible();
  });

  test('should remove completed items when clicked', async ({ page }) => {
    await page.getByRole('button', { name: 'Clear completed' }).click();
  });

  test('should be hidden when there are no items that are completed', async ({ page }) => {
    await page.getByRole('button', { name: 'Clear completed' }).click();
    await expect(page.getByRole('button', { name: 'Clear completed' })).toBeHidden();
  });
});

test.describe('Persistence', () => {
  test('should persist its data', async ({ page }) => {

    }


    // Ensure there is 1 completed item.

    // Now reload.
    await page.reload();
  });
});

test.describe('Routing', () => {
  test.beforeEach(async ({ page }) => {
    // before navigating to a new view, otherwise the items can get lost :(
    // in some frameworks like Durandal
  });

  test('should allow me to display active items', async ({ page }) => {

    await page.getByRole('link', { name: 'Active' }).click();
  });

  test('should respect the back button', async ({ page }) => {


    await test.step('Showing all items', async () => {
      await page.getByRole('link', { name: 'All' }).click();
    });

    await test.step('Showing active items', async () => {
      await page.getByRole('link', { name: 'Active' }).click();
    });

    await test.step('Showing completed items', async () => {
      await page.getByRole('link', { name: 'Completed' }).click();
    });

    await page.goBack();
    await page.goBack();
  });

  test('should allow me to display completed items', async ({ page }) => {
    await page.getByRole('link', { name: 'Completed' }).click();
  });

  test('should allow me to display all items', async ({ page }) => {
    await page.getByRole('link', { name: 'Active' }).click();
    await page.getByRole('link', { name: 'Completed' }).click();
    await page.getByRole('link', { name: 'All' }).click();
  });

  test('should highlight the currently applied filter', async ({ page }) => {
    await expect(page.getByRole('link', { name: 'All' })).toHaveClass('selected');
    
    //create locators for active and completed links
    const activeLink = page.getByRole('link', { name: 'Active' });
    const completedLink = page.getByRole('link', { name: 'Completed' });
    await activeLink.click();

    // Page change - active items.
    await expect(activeLink).toHaveClass('selected');
    await completedLink.click();

    // Page change - completed items.
    await expect(completedLink).toHaveClass('selected');
  });
});


  }
}

  return await page.waitForFunction(e => {
  }, expected);
}

  return await page.waitForFunction(e => {
  }, expected);
}

  return await page.waitForFunction(t => {
  }, title);
}
